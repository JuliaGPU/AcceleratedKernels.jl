include("tuning.jl")
include("slices.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")
include("radix_sort.jl")
include("bitonic_sort.jl")


"""
    sort!(
        v::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        workspace=nothing,
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

`workspace` takes the scratch memory of a [`workspace`](@ref) made for the same call, so that the
sort allocates none of its own.

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
    workspace=nothing,
)
    p, ord = sort_setup(v; backend, alg, dims, lt, by, rev, order)
    sort_run!(p.alg, one_based(v), p.backend, dims, ord, scratch_buffers(p, workspace, v);
              lt, by, rev, order)
    return v
end

plan(::typeof(sort!), v::AbstractArray; kwargs...) = first(sort_setup(v; kwargs...))

function sort_setup(
    v; backend=nothing, alg::Algorithm=Auto(), dims::Union{Colon, Integer}=Colon(),
    lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
)
    backend = resolve_backend(backend, v)
    v = one_based(v)
    ord = Base.Order.ord(lt, by, rev, order)
    a = resolve_sort(alg, backend, v, dims, ord)
    sizes, nested = sort_scratch(a, v, backend, dims, by)
    return Plan(backend, a, nested, sizes), ord
end

# The scratch of `sort!` with each algorithm, and the algorithms of the operations it calls
sort_scratch(a, v, backend, dims, by) = sort_sizes(a, v, backend, dims, by), (;)
sort_scratch(a::RadixSort, v, backend, dims, by) = radix_scratch(a, v, backend, dims)

function sort_sizes(a::MergeSort, v, backend, dims, by)
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return (;)
    by === identity || return (; keys=scratch(Base.promote_op(by, eltype(v)), size(v)),
                                 merge_by_key_scratch(a, layout, size(v),
                                                      Base.promote_op(by, eltype(v)),
                                                      size(v), eltype(v))...)
    return layout.len > 2 * a.block_size ? (; temp=scratch(eltype(v), size(v))) : (;)
end
sort_sizes(a::BitonicSort, v, backend, dims, by) = (;)
sort_sizes(a::CPUThreads.SampleSort, v, backend, dims, by) =
    dims isa Colon ? sample_sort_scratch(a, eltype(v), length(v)) : (;)

function sort_run!(a::MergeSort, v, backend, dims, ord, bufs; lt, by, rev, order)
    merge_sort!(v, backend, bufs; lt, by, rev, order, block_size=a.block_size, dims)
end

function sort_run!(a::RadixSort, v, backend, dims, ord, bufs; lt, by, rev, order)
    radix_sort!(v, backend, bufs; descending=ord === Base.Order.Reverse,
                block_size=a.block_size, items_per_thread=a.items_per_thread,
                chunk_size=a.chunk_size, dims)
end

function sort_run!(a::BitonicSort, v, backend, dims, ord, bufs; lt, by, rev, order)
    bitonic_sort!(v, backend; lt, by, rev, order, dims,
                  block_size=a.block_size, items_per_thread=a.items_per_thread)
end

function sort_run!(a::CPUThreads.SampleSort, v, backend, dims, ord, bufs; lt, by, rev, order)
    if dims isa Colon
        # `vec`: the local sorts use `Base.sort!`, which needs `dims` for other arrays
        sample_sort!(vec(v); lt, by, rev, order, max_tasks=a.max_tasks, min_elems=a.min_elems,
                     temp=get(bufs, :temp, nothing))
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
function sort(v::AbstractArray; backend::Union{Nothing, Backend}=nothing, workspace=nothing,
              kwargs...)
    backend = resolve_backend(backend, v)
    # The workspace must not alias the input either
    workspace === nothing || scratch_buffers(plan(sort!, v; backend, kwargs...), workspace, v)
    return sort!(copy_on(backend, v); backend, workspace, kwargs...)
end

plan(::typeof(sort), v::AbstractArray; kwargs...) = plan(sort!, v; kwargs...)


"""
    sortperm!(
        ix::AbstractArray,
        v::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        workspace=nothing,
    ) -> ix

Write into `ix` the stable permutation that sorts `v`, so that `v[ix]` is sorted; `ix` is always
overwritten. The keywords are those of [`sort!`](@ref).

With `dims=:`, `ix` needs as many elements as `v` and receives `v`'s linear indices
(`1:length(v)`, unless `v` is a vector whose axis starts elsewhere). With an integer `dims`, `ix`
must have the same axes as `v` and receives linear indices into `v`, so that `v[ix]` is sorted
along `dims`, like `Base.sortperm!(ix, A; dims)`.

`Auto()` chooses [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort) on the
host and [`MergeSort`](@ref) on GPUs; `MergeSort(lowmem=true)` avoids copying the keys.
`RadixSort` and `BitonicSort` have no permutation path. `backend` is derived from `ix` and `v`.
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
    workspace=nothing,
)
    check_sortperm_shape(ix, v, dims)
    sortperm_unchecked!(ix, v; backend, alg, dims, lt, by, rev, order, workspace)
end

plan(::typeof(sortperm!), ix::AbstractArray, v::AbstractArray; dims=Colon(), kwargs...) =
    (check_sortperm_shape(ix, v, dims); first(sortperm_setup(ix, v; dims, kwargs...)))

function check_sortperm_shape(ix, v, dims)
    if dims isa Colon
        length(ix) == length(v) || throw(ArgumentError(
            "index array must have as many elements as the input, $(length(ix)) != $(length(v))"))
    else
        axes(ix) == axes(v) || throw(ArgumentError(
            "index array must have the same axes as the input, $(axes(ix)) != $(axes(v))"))
    end
    nothing
end

# `sortperm!` without the check of `ix`'s shape
function sortperm_unchecked!(ix, v; backend=nothing, alg::Algorithm=Auto(),
                             dims::Union{Colon, Integer}=Colon(), lt=isless, by=identity,
                             rev::Union{Nothing, Bool}=nothing,
                             order::Base.Order.Ordering=Base.Order.Forward, workspace=nothing)
    p, ord = sortperm_setup(ix, v; backend, alg, dims, lt, by, rev, order)
    bufs = scratch_buffers(p, workspace, ix, v)
    ix_one_based = one_based(ix)
    sortperm_run!(p.alg, ix_one_based, one_based(v), p.backend, dims, ord, bufs;
                  lt, by, rev, order)
    # The kernels sort `v`'s linear indices counted from 1
    firstindex(v) == 1 || shift_indices!(ix_one_based, firstindex(v) - 1, p.backend)
    return ix
end

function shift_indices!(ix, shift, backend)
    foreachindex_on(Base.OneTo(length(ix)), backend) do i
        @inbounds ix[i] += shift
    end
end

function sortperm_setup(
    ix, v; backend=nothing, alg::Algorithm=Auto(), dims::Union{Colon, Integer}=Colon(),
    lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
)
    backend = resolve_backend(backend, ix, v)
    ix, v = one_based(ix), one_based(v)
    ord = Base.Order.ord(lt, by, rev, order)
    a = resolve_sort(alg, backend, v, dims, ord; perm=true)
    return Plan(backend, a, sortperm_scratch(a, ix, v, dims)), ord
end

# The scratch of `sortperm!` with each algorithm
function sortperm_scratch(a::MergeSort, ix, v, dims)
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return (;)
    a.lowmem && return layout.len > 2 * a.block_size ? (; temp=scratch(eltype(ix), size(ix))) : (;)
    # The keys are sorted in a copy, carrying the indices along
    return (; keys=scratch(eltype(v), size(v)),
            merge_by_key_scratch(a, layout, size(v), eltype(v), size(ix), eltype(ix))...)
end
sortperm_scratch(a::CPUThreads.SampleSort, ix, v, dims) =
    dims isa Colon ? sample_sort_scratch(a, eltype(ix), length(ix)) : (;)

function sortperm_run!(a::MergeSort, ix, v, backend, dims, ord, bufs; lt, by, rev, order)
    if a.lowmem
        merge_sortperm_lowmem!(ix, v, backend, bufs; lt, by, rev, order,
                               block_size=a.block_size, dims)
    else
        # Copies keys alongside indices, so comparisons never read global memory; the low-memory
        # path does two global loads per comparison, O(n log²n) global traffic at large n.
        merge_sortperm!(ix, v, backend, bufs; lt, by, rev, order, block_size=a.block_size, dims)
    end
end

function sortperm_run!(a::CPUThreads.SampleSort, ix, v, backend, dims, ord, bufs;
                       lt, by, rev, order)
    if dims isa Colon
        sample_sortperm!(vec(ix), vec(v); lt, by, rev, order,
                         max_tasks=a.max_tasks, min_elems=a.min_elems,
                         temp=get(bufs, :temp, nothing))
    else
        sample_sortperm_dims!(ix, v, ord, dims; max_tasks=a.max_tasks, min_elems=a.min_elems)
    end
end


"""
    sortperm(v::AbstractArray; kwargs...)

Out-of-place [`sortperm!`](@ref): return an `Int` array of `v`'s size, with axes starting at 1,
holding the permutation, with the same keywords.
"""
function sortperm(v::AbstractArray; backend::Union{Nothing, Backend}=nothing, kwargs...)
    backend = resolve_backend(backend, v)
    # (the result's axes start at 1, also where `v`'s do not)
    return sortperm_unchecked!(similar_on(backend, v, Int), v; backend, kwargs...)
end

# `ix` is only needed for its element type and shape, which `v` gives here
plan(::typeof(sortperm), v::AbstractArray; kwargs...) =
    first(sortperm_setup(IndexShape(v), v; kwargs...))

# The element type and shape of `similar(v, Int)`, without allocating it
struct IndexShape{N, A <: AbstractArray} <: AbstractArray{Int, N}
    v::A
end
IndexShape(v::AbstractArray{T, N}) where {T, N} = IndexShape{N, typeof(v)}(v)
Base.size(ix::IndexShape) = size(ix.v)
Base.axes(ix::IndexShape) = axes(ix.v)
backend_vote(::IndexShape) = nothing


"""
    sort_by_key!(
        keys::AbstractArray,
        values::AbstractArray;
        backend=nothing,
        alg::Algorithm=Auto(),
        dims::Union{Colon, Integer}=:,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        workspace=nothing,
    ) -> (keys, values)

Sort `keys` in place and apply the same permutation to `values`, stably: values with equal keys
keep their relative order. The ordering keywords and `dims` are those of [`sort!`](@ref); `values`
must have as many elements as `keys` (the same axes with an integer `dims`).

`Auto()` chooses [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort) on the
host and [`MergeSort`](@ref) on GPUs, the algorithms that support key/value sorting. `backend` is
derived from `keys` and `values`.

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
    workspace=nothing,
)
    p, ord = sort_by_key_setup(keys, values; backend, alg, dims, lt, by, rev, order)
    sort_by_key_run!(p.alg, one_based(keys), one_based(values), p.backend, dims, ord,
                     scratch_buffers(p, workspace, keys, values); lt, by, rev, order)
    return keys, values
end

plan(::typeof(sort_by_key!), keys::AbstractArray, values::AbstractArray; kwargs...) =
    first(sort_by_key_setup(keys, values; kwargs...))

function sort_by_key_setup(
    keys, values; backend=nothing, alg::Algorithm=Auto(), dims::Union{Colon, Integer}=Colon(),
    lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
)
    backend = resolve_backend(backend, keys, values)
    ord = Base.Order.ord(lt, by, rev, order)
    if dims isa Colon
        length(keys) == length(values) || throw(ArgumentError(
            "keys and values must have the same length, $(length(keys)) != $(length(values))"))
    else
        axes(keys) == axes(values) || throw(ArgumentError(
            "keys and values must have the same axes, $(axes(keys)) != $(axes(values))"))
    end
    keys, values = one_based(keys), one_based(values)
    a = resolve_sort(alg, backend, keys, dims, ord; pairs=true)
    return Plan(backend, a, sort_by_key_scratch(a, keys, values, dims)), ord
end

function sort_by_key_scratch(a::MergeSort, keys, values, dims)
    layout = slice_layout(keys, dims)
    (isempty(keys) || layout.len <= 1) && return (;)
    return merge_by_key_scratch(a, layout, size(keys), eltype(keys), size(values), eltype(values))
end
function sort_by_key_scratch(a::CPUThreads.SampleSort, keys, values, dims)
    # The sorting permutation (sorted by `sample_sortperm!` for `dims=:`), and the old keys and
    # values to gather from
    perm = dims isa Colon ? (; perm=sample_sort_scratch(a, Int, length(keys))) : (;)
    return (; ix=scratch(Int, size(keys)), perm..., keys=scratch(eltype(keys), size(keys)),
            values=scratch(eltype(values), size(values)))
end

function sort_by_key_run!(a::MergeSort, keys, values, backend, dims, ord, bufs;
                          lt, by, rev, order)
    merge_sort_by_key!(keys, values, backend, bufs; lt, by, rev, order, dims,
                       block_size=a.block_size)
end

function sort_by_key_run!(a::CPUThreads.SampleSort, keys, values, backend, dims, ord, bufs;
                          lt, by, rev, order)
    sample_sort_by_key!(keys, values, ord, dims, bufs; max_tasks=a.max_tasks,
                        min_elems=a.min_elems)
end
