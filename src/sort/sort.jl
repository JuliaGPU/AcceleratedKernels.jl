include("utils.jl")
include("slices.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")
include("radix_sort.jl")
include("bitonic_sort.jl")


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
    RadixSort(; block_size=nothing, items_per_thread=nothing)

GPU LSD radix sort for whole arrays of 32- and 64-bit integers and floats (`UInt32`, `Int32`,
`Float32`, `UInt64`, `Int64`, `Float64`) under the default ordering or its reverse; it is stable
and orders floats like `isless`. It does not support `dims`, custom `lt`/`by`, `sortperm!` or
`sort_by_key!`. `block_size` must be a power of two up to 1024, `items_per_thread` between 1 and
64. `items_per_thread` applies where radix sort's chunked kernels run, which needs atomics and a
`block_size` that is a multiple of 32 whose tiles fit local memory; elsewhere its portable kernels
sort one item per thread.
"""
Base.@kwdef struct RadixSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
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


include("tuning.jl")


"""
    sort!(
        v::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        temp::Union{Nothing, AbstractArray}=nothing,
    ) -> v

Sort `v` in place. `lt`, `by`, `rev` and `order` are those of `Base.sort!`, and so is the result:
by default the sort is stable. `dims=:` sorts the whole array as one vector (`Base.sort!` requires
`dims` for arrays of more than one dimension); an integer `dims` sorts each slice along that
dimension independently.

`alg` is [`Auto()`](@ref Auto) by default, which chooses an algorithm for the backend and device,
the element type, the ordering, and the length of the array or slices: on the host,
[`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort); on GPUs,
[`MergeSort`](@ref), or [`BitonicSort`](@ref) for short inputs or [`RadixSort`](@ref) for long
ones where the device's tuning enables them and they give the same result. `Auto(stable=false)`
also allows the unstable `BitonicSort` for elements that compare equal without being identical
(e.g. floats). An explicit algorithm, with any of its settings, is used as given or rejected with
an `ArgumentError`.

`backend` is derived from `v`; pass it only for arrays that do not determine their backend.

`temp` is an optional scratch array with the same length and element type as `v`, used by
`MergeSort`, `RadixSort` and, for `dims=:`, `CPUThreads.SampleSort`.

# Examples
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 100_000))
AK.sort!(v)                                         # Auto: stable, chosen for the device
AK.sort!(v; rev=true, alg=AK.Auto(stable=false))    # may use an unstable algorithm
AK.sort!(v; alg=AK.RadixSort(block_size=512))       # this algorithm, with this setting

A = CuArray(rand(Int32, 64, 10_000))
AK.sort!(A; dims=1)                                 # each column

AK.sort!(rand(1000))                                # host array: Julia threads
```
"""
function sort!(
    v::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    alg::Algorithm=Auto(),
    dims::Union{Colon, Integer}=Colon(),
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    backend = _resolve_backend(backend, v, temp)
    ord = Base.Order.ord(lt, by, rev, order)
    a = _resolve_sort(alg, backend, v, dims, ord)
    _sort_impl!(a, v, backend, dims, ord; lt, by, rev, order, temp)
    return v
end

function _sort_impl!(a::MergeSort, v, backend, dims, ord; lt, by, rev, order, temp)
    _merge_sort!(v, backend; lt, by, rev, order, block_size=a.block_size, temp, dims)
end

function _sort_impl!(a::RadixSort, v, backend, dims, ord; lt, by, rev, order, temp)
    _radix_sort!(v, backend; descending=ord === Base.Order.Reverse,
                 block_size=a.block_size, items_per_thread=a.items_per_thread, temp)
end

function _sort_impl!(a::BitonicSort, v, backend, dims, ord; lt, by, rev, order, temp)
    _bitonic_sort!(v, backend; lt, by, rev, order, dims,
                   block_size=a.block_size, items_per_thread=a.items_per_thread)
end

function _sort_impl!(a::CPUThreads.SampleSort, v, backend, dims, ord; lt, by, rev, order, temp)
    if dims isa Colon
        # `vec`: the local sorts use `Base.sort!`, which needs `dims` for other arrays
        _sample_sort!(vec(v); lt, by, rev, order, max_tasks=a.max_tasks, min_elems=a.min_elems, temp)
    else
        foreach_slice(v, dims; max_tasks=a.max_tasks, min_elems=a.min_elems) do slice
            Base.sort!(slice; order=ord)
        end
    end
end


"""
    sort(v::AbstractArray; kwargs...)

Out-of-place [`sort!`](@ref): sort a copy of `v`, with the same keywords.
"""
function sort(v::AbstractArray; backend::Union{Nothing, Backend}=nothing, kwargs...)
    backend = _resolve_backend(backend, v)
    return sort!(_copy(backend, v); backend, kwargs...)
end


"""
    sortperm!(
        ix::AbstractArray,
        v::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        temp::Union{Nothing, AbstractArray}=nothing,
    ) -> ix

Write into `ix` the stable permutation that sorts `v`, so that `v[ix]` is sorted; `ix` is always
overwritten. The keywords are those of [`sort!`](@ref).

With `dims=:`, `ix` needs as many elements as `v` and receives indices `1:length(v)`. With an
integer `dims`, `ix` must have the same axes as `v` and receives linear indices into `v`, so that
`v[ix]` is sorted along `dims`, like `Base.sortperm!(ix, A; dims)`.

`Auto()` chooses [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort) on the
host and [`MergeSort`](@ref) on GPUs; `MergeSort(lowmem=true)` avoids copying the keys.
`RadixSort` and `BitonicSort` have no permutation path. `backend` is derived from `ix` and `v`.
`temp` is an optional scratch array like `ix`.
"""
function sortperm!(
    ix::AbstractArray,
    v::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    alg::Algorithm=Auto(),
    dims::Union{Colon, Integer}=Colon(),
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    backend = _resolve_backend(backend, ix, v, temp)
    ord = Base.Order.ord(lt, by, rev, order)
    if dims isa Colon
        length(ix) == length(v) || throw(ArgumentError(
            "index array must have as many elements as the input, $(length(ix)) != $(length(v))"))
    else
        axes(ix) == axes(v) || throw(ArgumentError(
            "index array must have the same axes as the input, $(axes(ix)) != $(axes(v))"))
    end
    a = _resolve_sort(alg, backend, v, dims, ord; perm=true)
    _sortperm_impl!(a, ix, v, backend, dims, ord; lt, by, rev, order, temp)
    return ix
end

function _sortperm_impl!(a::MergeSort, ix, v, backend, dims, ord; lt, by, rev, order, temp)
    if a.lowmem
        _merge_sortperm_lowmem!(ix, v, backend; lt, by, rev, order,
                                block_size=a.block_size, temp, dims)
    else
        # Copies keys alongside indices, so comparisons never read global memory; the low-memory
        # path does two global loads per comparison, O(n log²n) global traffic at large n.
        _merge_sortperm!(ix, v, backend; lt, by, rev, order,
                         block_size=a.block_size, temp_ix=temp, dims)
    end
end

function _sortperm_impl!(a::CPUThreads.SampleSort, ix, v, backend, dims, ord;
                         lt, by, rev, order, temp)
    if dims isa Colon
        _sample_sortperm!(vec(ix), vec(v); lt, by, rev, order,
                          max_tasks=a.max_tasks, min_elems=a.min_elems, temp)
    else
        _sample_sortperm_dims!(ix, v, ord, dims; max_tasks=a.max_tasks, min_elems=a.min_elems)
    end
end


"""
    sortperm(v::AbstractArray; kwargs...)

Out-of-place [`sortperm!`](@ref): return an `Int` array shaped like `v` holding the permutation,
with the same keywords.
"""
function sortperm(v::AbstractArray; backend::Union{Nothing, Backend}=nothing, kwargs...)
    backend = _resolve_backend(backend, v)
    return sortperm!(_similar(backend, v, Int), v; backend, kwargs...)
end


"""
    sort_by_key!(
        keys::AbstractArray,
        values::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        temp_keys::Union{Nothing, AbstractArray}=nothing,
        temp_values::Union{Nothing, AbstractArray}=nothing,
    ) -> (keys, values)

Sort `keys` in place and apply the same permutation to `values`, stably: values with equal keys
keep their relative order. The ordering keywords and `dims` are those of [`sort!`](@ref); `values`
must have as many elements as `keys` (the same axes with an integer `dims`).

`Auto()` chooses [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort) on the
host and [`MergeSort`](@ref) on GPUs, the algorithms that support key/value sorting. `backend` is
derived from `keys` and `values`. `temp_keys` and `temp_values` are optional scratch arrays like
`keys` and `values`.

Thrust, oneDPL and Kokkos call this operation `sort_by_key`, CUB `SortPairs`.

# Examples
```julia
import AcceleratedKernels as AK
using Metal

keys = MtlArray(rand(Int32(1):Int32(10), 1000))
values = MtlArray(Int32.(1:1000))
AK.sort_by_key!(keys, values)      # values of equal keys stay in ascending order
```
"""
function sort_by_key!(
    keys::AbstractArray,
    values::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    alg::Algorithm=Auto(),
    dims::Union{Colon, Integer}=Colon(),
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    temp_keys::Union{Nothing, AbstractArray}=nothing,
    temp_values::Union{Nothing, AbstractArray}=nothing,
)
    backend = _resolve_backend(backend, keys, values, temp_keys, temp_values)
    ord = Base.Order.ord(lt, by, rev, order)
    if dims isa Colon
        length(keys) == length(values) || throw(ArgumentError(
            "keys and values must have the same length, $(length(keys)) != $(length(values))"))
    else
        axes(keys) == axes(values) || throw(ArgumentError(
            "keys and values must have the same axes, $(axes(keys)) != $(axes(values))"))
    end
    a = _resolve_sort(alg, backend, keys, dims, ord; pairs=true)
    _sort_by_key_impl!(a, keys, values, backend, dims, ord;
                       lt, by, rev, order, temp_keys, temp_values)
    return keys, values
end

function _sort_by_key_impl!(a::MergeSort, keys, values, backend, dims, ord;
                            lt, by, rev, order, temp_keys, temp_values)
    _merge_sort_by_key!(keys, values, backend; lt, by, rev, order, dims,
                        block_size=a.block_size, temp_keys, temp_values)
end

function _sort_by_key_impl!(a::CPUThreads.SampleSort, keys, values, backend, dims, ord;
                            lt, by, rev, order, temp_keys, temp_values)
    _sample_sort_by_key!(keys, values, ord, dims;
                         max_tasks=a.max_tasks, min_elems=a.min_elems, temp_keys, temp_values)
end
