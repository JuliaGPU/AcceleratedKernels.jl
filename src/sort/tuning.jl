# Selection and tuning of sorting algorithms.
#
# Every sorting entry point resolves its `alg` once, before touching any data:
#   1. `check_fields` checks the fields the caller set explicitly;
#   2. `select_sort` picks an algorithm for `Auto`, from the device's `SortTuning`, the
#      capabilities of the backend and the call's static facts (element type, layout, ordering);
#   3. `fill_defaults` fills the fields left at `nothing` from the tuning;
#   4. `check_sort` checks the complete algorithm against the backend's capabilities, the operation
#      and the arguments, throwing an `ArgumentError` for anything it cannot run.
# Selection returns algorithms with unset fields, so `Auto` and explicit algorithms are filled and
# checked by the same code.


# Sorting algorithms (`CPUThreads.SampleSort` is defined with the other CPUThreads algorithms)

"""
    MergeSort(; block_size=nothing, lowmem=false)

GPU merge sort: stable, for every element type and ordering, whole arrays and `dims`, and the only
kernel algorithm for `sortperm!` and [`sort_by_key!`](@ref). Each block of `block_size` threads
(any positive number) sorts a tile of `2 * block_size` elements in local memory, then global
passes merge the tiles. `lowmem=true` selects a `sortperm!` path that does not copy the keys, at
the cost of reading them from global memory in every comparison; other operations reject it.
"""
Base.@kwdef struct MergeSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    lowmem::Bool = false
end

"""
    RadixSort(; block_size=nothing, items_per_thread=nothing, chunk_size=nothing)

GPU LSD radix sort for 32- and 64-bit integers and floats (`UInt32`, `Int32`, `Float32`,
`UInt64`, `Int64`, `Float64`) under the default ordering or its reverse, for whole arrays and
`dims`; it is stable and orders floats like `isless`. It does not support custom `lt`/`by`,
`sortperm!` or `sort_by_key!`, nor arrays or slices of more than 2^32 elements. `block_size`
must be a power of two up to 1024, `items_per_thread` between 1 and 64. `items_per_thread` and
`chunk_size` apply where radix sort's chunked kernels run, which needs atomics, a `block_size`
that is a multiple of `chunk_size`, and tiles that fit local memory; elsewhere its portable
kernels sort one item per thread. The chunked kernels rank each element among the earlier
elements of its chunk: `chunk_size`, a power of two up to 1024, trades those comparisons against
the chunk histograms in local memory.

Along `dims`, each pass sorts all slices at once. It pays off for long slices: every slice takes
at least one block of the passes, so short ones leave most threads idle, and slices along other
dimensions than the first are read with a stride. Where the chunked kernels run and a slice of up
to `2 * block_size` elements fits in local memory, one block sorts each slice there instead.
"""
Base.@kwdef struct RadixSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
    chunk_size::Union{Nothing, Int} = nothing
end

"""
    BitonicSort(; block_size=nothing, items_per_thread=nothing)

GPU bitonic sorting network, for whole arrays and `dims`, every element type and ordering. It is
unstable and does not support `sortperm!` or `sort_by_key!`. Fastest for small arrays and short
slices: tiles of `block_size * items_per_thread` elements sort in local memory, larger inputs need
global passes. Both settings must be positive powers of two.
"""
Base.@kwdef struct BitonicSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end


"""
    SortTuning(; kwargs...)

Values that drive [`Auto`](@ref) selection and fill unset algorithm fields for sorting on one
device, as returned by [`sort_tuning`](@ref).

- `bitonic_max_len`: `Auto` picks `BitonicSort` for slices and arrays up to this length, where
  its instability is allowed (see [`Auto`](@ref)).
- `radix_min_len`: `Auto` picks `RadixSort` for whole-array sorts from this length, for the
  element types and orderings it supports.
- `merge_block_size`, `radix_block_size`, `radix_items_per_thread`, `radix_chunk_size`,
  `bitonic_block_size`, `bitonic_items_per_thread`: settings for the algorithms' unset fields.
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
    radix_chunk_size::Int = 32
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


# The operation an algorithm is resolved for, for error messages
_sort_opname(perm, pairs) = pairs ? "sort_by_key!" : perm ? "sortperm!" : "sort!"


"""
    resolve_sort(alg, backend, v, dims, ord; perm=false, pairs=false) -> SortAlgorithm

Resolve `alg` for sorting `v` (the keys) along `dims` under ordering `ord` on `backend`, for
`sort!` (`perm == pairs == false`), `sortperm!` (`perm`) or `sort_by_key!` (`pairs`). Returns a
concrete algorithm with every field set, or throws an `ArgumentError`.
"""
function resolve_sort(alg::Algorithm, backend::Backend, v::AbstractArray, dims,
                       ord::Base.Order.Ordering; perm::Bool=false, pairs::Bool=false)
    T = eltype(v)
    layout = slice_layout(v, dims)
    check_fields(alg)
    t = sort_tuning(backend, T)
    a = alg isa Auto ? select_sort(alg, backend, t, T, layout, ord; perm, pairs) : alg
    a = fill_defaults(a, t, T)
    check_sort(a, backend, T, layout, ord; perm, pairs)
    return a
end

function select_sort(a::Auto, backend, t::SortTuning, ::Type{T}, layout, ord;
                      perm, pairs) where {T}
    runs_threads(backend) && return CPUThreads.SampleSort()
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

check_fields(a::MergeSort) = check_positive(a, :block_size)

function check_fields(a::RadixSort)
    # Bounded, so that the local-memory footprints computed from them cannot overflow
    check_pow2(a, :block_size)
    check_positive(a, :items_per_thread)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "RadixSort: `block_size` must be at most 1024, got $(a.block_size)"))
    a.items_per_thread === nothing || a.items_per_thread <= 64 || throw(ArgumentError(
        "RadixSort: `items_per_thread` must be at most 64, got $(a.items_per_thread)"))
    check_pow2(a, :chunk_size)
    a.chunk_size === nothing || a.chunk_size <= 1024 || throw(ArgumentError(
        "RadixSort: `chunk_size` must be at most 1024, got $(a.chunk_size)"))
    nothing
end

function check_fields(a::BitonicSort)
    check_pow2(a, :block_size)
    check_pow2(a, :items_per_thread)
    if a.block_size !== nothing && a.items_per_thread !== nothing
        a.block_size <= typemax(Int) ÷ a.items_per_thread || throw(ArgumentError(
            "BitonicSort: `block_size * items_per_thread` overflows " *
            "($(a.block_size) * $(a.items_per_thread))"))
    end
    nothing
end

check_fields(a::CPUThreads.SampleSort) = check_threads(a)


# Fill unset fields from the tuning

fill_defaults(a::MergeSort, t::SortTuning, T) =
    MergeSort(something(a.block_size, t.merge_block_size), a.lowmem)
fill_defaults(a::RadixSort, t::SortTuning, T) =
    RadixSort(something(a.block_size, t.radix_block_size),
              something(a.items_per_thread, t.radix_items_per_thread),
              something(a.chunk_size, t.radix_chunk_size))
fill_defaults(a::BitonicSort, t::SortTuning, T) =
    BitonicSort(something(a.block_size, t.bitonic_block_size),
                something(a.items_per_thread, t.bitonic_items_per_thread))
fill_defaults(a::CPUThreads.SampleSort, t::SortTuning, T) = fill_threads(a, t)
fill_defaults(a::Algorithm, t::SortTuning, T) =
    throw(ArgumentError("$(algname(a)) is not a sorting algorithm"))


# Check a complete algorithm against the backend, the operation and the arguments

function check_sort(a::MergeSort, backend, T, layout, ord; perm, pairs)
    check_fields(a)
    require_kernels(a, backend)
    !a.lowmem || (perm && !pairs) || throw(ArgumentError(
        "MergeSort(lowmem=true) is only supported by sortperm!, not by $(_sort_opname(perm, pairs))"))
    nothing
end

function check_sort(a::RadixSort, backend, ::Type{T}, layout, ord; perm, pairs) where {T}
    check_fields(a)
    require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "RadixSort does not support $(_sort_opname(perm, pairs))"))
    _rs_supported(T) || throw(ArgumentError(
        "RadixSort does not support element type $T; it supports 32- and 64-bit integers " *
        "and floats"))
    _rs_ordering(ord) || throw(ArgumentError(
        "RadixSort only supports the default ordering and its reverse (no custom `lt` or `by`)"))
    layout.len <= _RS_MAX_LEN || throw(ArgumentError(
        "RadixSort sorts at most 2^32 elements per array or slice, got $(layout.len)"))
    _rs_portable_local_memory(T, a.block_size) <= LOCAL_MEMORY_BUDGET || throw(ArgumentError(
        "RadixSort: block_size=$(a.block_size) needs more than $(LOCAL_MEMORY_BUDGET) bytes " *
        "of local memory for element type $T"))
    nothing
end

function check_sort(a::BitonicSort, backend, T, layout, ord; perm, pairs)
    check_fields(a)
    require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "BitonicSort is unstable and does not support $(_sort_opname(perm, pairs))"))
    nothing
end

function check_sort(a::CPUThreads.SampleSort, backend, T, layout, ord; perm, pairs)
    check_fields(a)
    require_threads(a, backend)
end
