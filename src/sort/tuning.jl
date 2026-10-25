# Selection and tuning of sorting algorithms.
#
# Every sorting entry point resolves its `alg` once, before touching any data:
#   1. `select_sort` picks an algorithm for `Auto`, from the device's `SortTuning`, the
#      capabilities of the backend and the call's static facts (element type, layout, ordering);
#   2. `fill_defaults` fills the fields left at `nothing` from the tuning;
#   3. `check_sort` checks the complete algorithm (its fields with `check_fields`) against the
#      backend's capabilities, the operation and the arguments, throwing an `ArgumentError` for
#      anything it cannot run.
# Selection returns algorithms with unset fields, so `Auto` and explicit algorithms are filled and
# checked by the same code.


# Sorting algorithms (`CPUThreads.SampleSort` is defined with the other CPUThreads algorithms)

"""
    MergeSort(; block_size=nothing, lowmem=false)

GPU merge sort: stable, for every element type and ordering, whole arrays and `dims`, and the only
kernel algorithm for `sortperm!` and [`sort_by_key!`](@ref). Each block of `block_size` threads
(any positive number) sorts a tile of `2 * block_size` elements in local memory, then global
passes merge the tiles. A tile, with the values moved along, must fit in 32 KiB; for large
elements, the default `block_size` shrinks to fit. `lowmem=true` selects a `sortperm!` path that
does not copy the keys, at the cost of reading them from global memory in every comparison; other
operations reject it.
"""
Base.@kwdef struct MergeSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    lowmem::Bool = false
end

"""
    RadixSort(; block_size=nothing, items_per_thread=nothing, chunk_size=nothing)

GPU LSD radix sort: stable, for `UInt32`, `Int32`, `Float32`, `UInt64`, `Int64` and `Float64`
under the default ordering or its reverse (floats ordered like `isless`), on whole arrays and
along `dims`, with at most 2^32 elements per array or slice. It supports neither custom
`lt`/`by` nor `sortperm!` or `sort_by_key!`.

`block_size` is a power of two up to 1024. Where the backend has atomics, `chunk_size` (a power
of two up to 1024) divides `block_size` and the tile fits local memory, each thread sorts
`items_per_thread` (1 to 64) elements, ranking each among the earlier elements of its chunk:
`chunk_size` trades those comparisons against the chunks' histograms in local memory. Elsewhere
each thread sorts one element.

Along `dims`, each pass sorts all slices at once. It pays off for long slices: every slice takes
at least one block of the passes, so short ones leave most threads idle, and slices along other
dimensions than the first are read with a stride. Where each thread sorts `items_per_thread`
elements and a slice of up to `2 * block_size` elements fits in local memory, one block sorts
each slice there instead.
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
global passes. Both settings must be positive powers of two, and a tile must fit in 32 KiB.
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

The defaults never pick `BitonicSort` or `RadixSort`, so GPUs sort with `MergeSort` where a
backend's tuning does not set these thresholds.
"""
Base.@kwdef struct SortTuning
    bitonic_max_len::Int = 0
    radix_min_len::Int = typemax(Int)
    merge_block_size::Int = 256
    # Tiles of 1024 elements ranked in chunks of 64: sorting 1M or 16M Float32 or 64-bit integers
    # takes 5-13% less time than with 256 threads and chunks of 32 on an RTX 5080, 9-15% on an
    # RTX 6000 Ada, 6-24% on an M1, 39-44% on an Arc A770 and 19-52% on a Radeon iGPU (gfx1036);
    # the fastest setting measured on each of them is at most 12% faster.
    radix_block_size::Int = 512
    radix_items_per_thread::Int = 2
    radix_chunk_size::Int = 64
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
sort_opname(perm, pairs) = pairs ? "sort_by_key!" : perm ? "sortperm!" : "sort!"


"""
    resolve_sort(alg, backend, v, dims, ord; perm=false, pairs=false,
                 merge_eltypes=(eltype(v), Nothing)) -> SortAlgorithm

Resolve `alg` for sorting `v` (the keys) along `dims` under ordering `ord` on `backend`, for
`sort!` (`perm == pairs == false`), `sortperm!` (`perm`) or `sort_by_key!` (`pairs`). Returns a
concrete algorithm with every field set, or throws an `ArgumentError`. `merge_eltypes` are the
element types `MergeSort` holds in local memory: the keys it compares (`by.(v)` with a `by`) and
the values it moves along with them (`Nothing` for none).
"""
function resolve_sort(alg::Algorithm, backend::Backend, v::AbstractArray, dims,
                      ord::Base.Order.Ordering; perm::Bool=false, pairs::Bool=false,
                      merge_eltypes::Tuple{Type, Type}=(eltype(v), Nothing))
    T = eltype(v)
    layout = slice_layout(v, dims)
    t = sort_tuning(backend, T)
    a = alg isa Auto ? select_sort(alg, backend, t, T, layout, ord; perm, pairs) : alg
    a = fill_defaults(a, t, T, merge_eltypes)
    check_sort(a, backend, T, layout, ord; perm, pairs, merge_eltypes)
    return a
end

function select_sort(a::Auto, backend, t::SortTuning, ::Type{T}, layout, ord;
                     perm, pairs) where {T}
    runs_threads(backend) && return CPUThreads.SampleSort()
    (perm || pairs) && return MergeSort()   # the only stable, key/value-capable kernel algorithm
    if (!a.stable || ties_identical(T, ord)) && layout.len <= t.bitonic_max_len &&
       bitonic_fits(T, t.bitonic_block_size, t.bitonic_items_per_thread)
        return BitonicSort()
    elseif layout isa FlatLayout && radix_supports_eltype(T) && radix_supports_ordering(ord) &&
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
ties_identical(::Type{T}, ord) where {T} =
    isconcretetype(T) && T <: Union{Base.BitInteger, Bool, Char} &&
    ord isa Union{Base.Order.ForwardOrdering, Base.Order.ReverseOrdering{Base.Order.ForwardOrdering}}

# The orderings radix sort supports: the default one and its reverse.
radix_supports_ordering(ord) = ord === Base.Order.Forward || ord === Base.Order.Reverse


# Domain checks of the fields

check_fields(a::MergeSort) = check_positive(a, :block_size)

function check_fields(a::RadixSort)
    # Bounded, so that the local-memory footprints computed from them cannot overflow
    check_block_size(a)
    check_positive(a, :items_per_thread)
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

# The largest block size whose tiles of `2 * block_size` keys and values fit in local memory.
# With `lowmem`, the tiles hold only the values: the indices of `sortperm!`, compared through the
# keys in global memory.
function merge_max_block(lowmem, (K, V))
    bytes = lowmem ? local_sizeof(V) : local_sizeof(K) + local_sizeof(V)
    LOCAL_MEMORY_BUDGET ÷ max(2 * bytes, 1)
end

# The default block size shrinks to a power of two whose tiles fit in local memory
function fill_defaults(a::MergeSort, t::SortTuning, T, merge_eltypes)
    fitting = prevpow(2, max(merge_max_block(a.lowmem, merge_eltypes), 1))
    MergeSort(something(a.block_size, min(t.merge_block_size, fitting)), a.lowmem)
end
fill_defaults(a::RadixSort, t::SortTuning, T, _) =
    RadixSort(something(a.block_size, t.radix_block_size),
              something(a.items_per_thread, t.radix_items_per_thread),
              something(a.chunk_size, t.radix_chunk_size))
fill_defaults(a::BitonicSort, t::SortTuning, T, _) =
    BitonicSort(something(a.block_size, t.bitonic_block_size),
                something(a.items_per_thread, t.bitonic_items_per_thread))
fill_defaults(a::CPUThreads.SampleSort, t::SortTuning, T, _) = fill_threads(a, t)
fill_defaults(a::Algorithm, t::SortTuning, T, _) =
    throw(ArgumentError("$(algname(a)) is not a sorting algorithm"))


# Check a complete algorithm against the backend, the operation and the arguments

function check_sort(a::MergeSort, backend, T, layout, ord; perm, pairs, merge_eltypes)
    check_fields(a)
    require_kernels(a, backend)
    !a.lowmem || (perm && !pairs) || throw(ArgumentError(
        "MergeSort(lowmem=true) is only supported by sortperm!, not by $(sort_opname(perm, pairs))"))
    K, V = merge_eltypes
    a.block_size <= merge_max_block(a.lowmem, merge_eltypes) || throw(ArgumentError(
        "MergeSort: block_size=$(a.block_size) needs more than $(LOCAL_MEMORY_BUDGET) bytes of " *
        "local memory for keys of type $K" * (V === Nothing ? "" : " and values of type $V")))
    nothing
end

function check_sort(a::RadixSort, backend, ::Type{T}, layout, ord; perm, pairs, _...) where {T}
    check_fields(a)
    require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "RadixSort does not support $(sort_opname(perm, pairs))"))
    radix_supports_eltype(T) || throw(ArgumentError(
        "RadixSort does not support element type $T; it supports 32- and 64-bit integers " *
        "and floats"))
    radix_supports_ordering(ord) || throw(ArgumentError(
        "RadixSort only supports the default ordering and its reverse (no custom `lt` or `by`)"))
    layout.len <= RADIX_MAX_LEN || throw(ArgumentError(
        "RadixSort sorts at most 2^32 elements per array or slice, got $(layout.len)"))
    radix_portable_localmem(T, a.block_size) <= LOCAL_MEMORY_BUDGET || throw(ArgumentError(
        "RadixSort: block_size=$(a.block_size) needs more than $(LOCAL_MEMORY_BUDGET) bytes " *
        "of local memory for element type $T"))
    nothing
end

function check_sort(a::BitonicSort, backend, T, layout, ord; perm, pairs, _...)
    check_fields(a)
    require_kernels(a, backend)
    (perm || pairs) && throw(ArgumentError(
        "BitonicSort is unstable and does not support $(sort_opname(perm, pairs))"))
    bitonic_fits(T, a.block_size, a.items_per_thread) || throw(ArgumentError(
        "BitonicSort: tiles of block_size * items_per_thread = " *
        "$(a.block_size * a.items_per_thread) elements need more than $(LOCAL_MEMORY_BUDGET) " *
        "bytes of local memory for element type $T"))
    nothing
end

function check_sort(a::CPUThreads.SampleSort, backend, T, layout, ord; perm, pairs, _...)
    check_fields(a)
    require_threads(a, backend)
end
