# Selection and tuning of sorting algorithms.
#
# Every sorting entry point resolves its `alg` once, before touching any data:
#   1. `_checkdomain` checks the fields the caller set explicitly;
#   2. `_select_sort` picks an algorithm for `Auto`, from the device's `SortTuning`, the
#      capabilities of the backend and the call's static facts (element type, layout, ordering);
#   3. `_fill` fills the fields left at `nothing` from the tuning;
#   4. `_check` checks the complete algorithm against the backend's capabilities, the operation
#      and the arguments, throwing an `ArgumentError` for anything it cannot run.
# Selection returns algorithms with unset fields, so `Auto` and explicit algorithms are filled and
# checked by the same code.


"""
    SortTuning(; kwargs...)

Values that drive [`Auto`](@ref) selection and fill unset algorithm fields for sorting on one
device, as returned by [`sort_tuning`](@ref).

- `bitonic_max_len`: `Auto` picks `BitonicSort` for slices and arrays up to this length, where
  its instability is allowed (see [`Auto`](@ref)).
- `radix_min_len`: `Auto` picks `RadixSort` for whole-array sorts from this length, for the
  element types and orderings it supports.
- `merge_block_size`, `radix_block_size`, `radix_items_per_thread`, `bitonic_block_size`,
  `bitonic_items_per_thread`: settings for the algorithms' unset fields.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.SampleSort`; `max_tasks` defaults
  to `Threads.nthreads()`.

The defaults never pick `BitonicSort` or `RadixSort`, and reproduce AK's historical settings.
Internal: the fields may change in any release.
"""
Base.@kwdef struct SortTuning
    bitonic_max_len::Int = 0
    radix_min_len::Int = typemax(Int)
    merge_block_size::Int = 256
    radix_block_size::Int = 256
    radix_items_per_thread::Int = 2
    bitonic_block_size::Int = 256
    bitonic_items_per_thread::Int = 8
    threads_min_elems::Int = 1
end

"""
    sort_tuning(backend, T) -> SortTuning

The sorting tuning for element type `T` on `backend`'s current device (the device the calling
task would launch on). AK defines this generic method; AK's package extensions add one method
per backend type, which may choose different values per device.
"""
sort_tuning(::Backend, ::Type) = SortTuning()


# Algorithm names in error messages
_algname(a) = nameof(typeof(a))
_algname(::CPUThreads.SampleSort) = "CPUThreads.SampleSort"


# The operation an algorithm is resolved for, for error messages
_sort_opname(perm, pairs) = pairs ? "sort_by_key!" : perm ? "sortperm!" : "sort!"


"""
    _resolve_sort(alg, backend, v, dims, ord; perm=false, pairs=false) -> SortAlgorithm

Resolve `alg` for sorting `v` (the keys) along `dims` under ordering `ord` on `backend`, for
`sort!` (`perm == pairs == false`), `sortperm!` (`perm`) or `sort_by_key!` (`pairs`). Returns a
concrete algorithm with every field set, or throws an `ArgumentError`.
"""
function _resolve_sort(alg::Algorithm, backend::Backend, v::AbstractArray, dims,
                       ord::Base.Order.Ordering; perm::Bool=false, pairs::Bool=false)
    T = eltype(v)
    layout = slice_layout(v, dims)
    _checkdomain(alg)
    t = sort_tuning(backend, T)
    a = alg isa Auto ? _select_sort(alg, backend, t, T, layout, ord; perm, pairs) : alg
    a = _fill(a, t, T)
    _check(a, backend, T, layout, ord; perm, pairs)
    return a
end

function _select_sort(a::Auto, backend, t::SortTuning, ::Type{T}, layout, ord;
                      perm, pairs) where {T}
    _runs_threads(backend) && return CPUThreads.SampleSort()
    (perm || pairs) && return MergeSort()   # the only stable, key/value-capable kernel algorithm
    if (!a.stable || _ties_invisible(T, ord)) && layout.len <= t.bitonic_max_len
        return BitonicSort()
    elseif layout isa FlatLayout && _rs_supported(T) && _rs_ordering(ord) &&
           layout.len >= t.radix_min_len
        return RadixSort()                  # whole arrays only; a single slice is not flat
    end
    return MergeSort()
end

# Equal elements under `ord` are bitwise identical, so an unstable sort's result equals a
# stable sort's. Floats are excluded: `isless` treats NaNs with different bit patterns as equal.
# `Base.Order.ord(isless, identity, rev, Forward)` is one of the two orderings accepted here;
# custom `lt`/`by` give other ordering types. `isconcretetype` excludes union element types,
# whose equal values can differ in type.
_ties_invisible(::Type{T}, ord) where {T} =
    isconcretetype(T) && T <: Union{Base.BitInteger, Bool, Char} &&
    ord isa Union{Base.Order.ForwardOrdering, Base.Order.ReverseOrdering{Base.Order.ForwardOrdering}}

# The orderings radix sort supports: the default one and its reverse.
_rs_ordering(ord) = ord === Base.Order.Forward || ord === Base.Order.Reverse


# Domain checks of explicitly set fields, before any arithmetic uses them

_checkdomain(::Algorithm) = nothing

function _check_positive(a, field)
    x = getfield(a, field)
    x === nothing || x >= 1 ||
        throw(ArgumentError("$(_algname(a)): `$field` must be positive, got $x"))
    nothing
end

function _check_pow2(a, field)
    x = getfield(a, field)
    x === nothing || (x >= 1 && ispow2(x)) ||
        throw(ArgumentError("$(_algname(a)): `$field` must be a positive power of two, got $x"))
    nothing
end

_checkdomain(a::MergeSort) = _check_positive(a, :block_size)

function _checkdomain(a::RadixSort)
    # Bounded, so that the local-memory footprints computed from them cannot overflow
    _check_pow2(a, :block_size)
    _check_positive(a, :items_per_thread)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "RadixSort: `block_size` must be at most 1024, got $(a.block_size)"))
    a.items_per_thread === nothing || a.items_per_thread <= 64 || throw(ArgumentError(
        "RadixSort: `items_per_thread` must be at most 64, got $(a.items_per_thread)"))
    nothing
end

function _checkdomain(a::BitonicSort)
    _check_pow2(a, :block_size)
    _check_pow2(a, :items_per_thread)
    if a.block_size !== nothing && a.items_per_thread !== nothing
        a.block_size <= typemax(Int) ÷ a.items_per_thread || throw(ArgumentError(
            "BitonicSort: `block_size * items_per_thread` overflows " *
            "($(a.block_size) * $(a.items_per_thread))"))
    end
    nothing
end

function _checkdomain(a::CPUThreads.SampleSort)
    _check_positive(a, :max_tasks)
    _check_positive(a, :min_elems)
end


# Fill unset fields from the tuning

_fill(a::MergeSort, t::SortTuning, T) =
    MergeSort(something(a.block_size, t.merge_block_size), a.lowmem)
_fill(a::RadixSort, t::SortTuning, T) =
    RadixSort(something(a.block_size, t.radix_block_size),
              something(a.items_per_thread, t.radix_items_per_thread))
_fill(a::BitonicSort, t::SortTuning, T) =
    BitonicSort(something(a.block_size, t.bitonic_block_size),
                something(a.items_per_thread, t.bitonic_items_per_thread))
_fill(a::CPUThreads.SampleSort, t::SortTuning, T) =
    CPUThreads.SampleSort(something(a.max_tasks, Threads.nthreads()),
                          something(a.min_elems, t.threads_min_elems))
_fill(a::Algorithm, t::SortTuning, T) =
    throw(ArgumentError("$(_algname(a)) is not a sorting algorithm"))


# Check a complete algorithm against the backend, the operation and the arguments

function _require_kernels(a, backend)
    _runs_kernels(backend) || throw(ArgumentError(
        "$(_algname(a)) runs AcceleratedKernels' GPU kernels, which " *
        "$(_backend_name(backend)) cannot run (on the host, this needs KernelAbstractions 0.10); " *
        "use `Auto()` or a `CPUThreads` algorithm"))
    nothing
end

function _check(a::MergeSort, backend, T, layout, ord; perm, pairs)
    _checkdomain(a)
    _require_kernels(a, backend)
    !a.lowmem || (perm && !pairs) || throw(ArgumentError(
        "MergeSort(lowmem=true) is only supported by sortperm!, not by $(_sort_opname(perm, pairs))"))
    nothing
end

function _check(a::RadixSort, backend, ::Type{T}, layout, ord; perm, pairs) where {T}
    _checkdomain(a)
    _require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "RadixSort does not support $(_sort_opname(perm, pairs))"))
    layout isa FlatLayout || throw(ArgumentError(
        "RadixSort does not support sorting along `dims`"))
    _rs_supported(T) || throw(ArgumentError(
        "RadixSort does not support element type $T; it supports 32- and 64-bit integers " *
        "and floats"))
    _rs_ordering(ord) || throw(ArgumentError(
        "RadixSort only supports the default ordering and its reverse (no custom `lt` or `by`)"))
    _rs_portable_local_memory(T, a.block_size) <= LOCAL_MEMORY_BUDGET || throw(ArgumentError(
        "RadixSort: block_size=$(a.block_size) needs more than $(LOCAL_MEMORY_BUDGET) bytes " *
        "of local memory for element type $T"))
    nothing
end

function _check(a::BitonicSort, backend, T, layout, ord; perm, pairs)
    _checkdomain(a)
    _require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "BitonicSort is unstable and does not support $(_sort_opname(perm, pairs))"))
    nothing
end

function _check(a::CPUThreads.SampleSort, backend, T, layout, ord; perm, pairs)
    _checkdomain(a)
    _runs_threads(backend) || throw(ArgumentError(
        "CPUThreads.SampleSort only runs on the host backend, not on $(_backend_name(backend))"))
    nothing
end
