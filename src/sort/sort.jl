include("tuning.jl")
include("slices.jl")
include("merge_sort.jl")
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
[`BitonicSort`](@ref) for short arrays and slices, [`RadixSort`](@ref) for long ones of the
element types and orderings it supports, and [`MergeSort`](@ref) otherwise, with lengths measured
for each backend (`MergeSort` on backends without such measurements). `BitonicSort` is unstable,
so by default `Auto` uses it only where equal elements are identical (integers, `Bool`, `Char`
under the default ordering or its reverse); `Auto(stable=false)` also allows it for others
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
    lt::L=isless,
    by::B=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    workspace=nothing,
) where {L, B}
    s = sort_setup(v; backend, alg, dims, lt, by, rev, order)
    p = s.plan
    sort_run!(p.alg, one_based(v), p.backend, dims, s.ord, by, scratch_buffers(p, workspace, v))
    return v
end

plan(::typeof(sort!), v::AbstractArray; kwargs...) = sort_setup(v; kwargs...).plan

function sort_setup(
    v; backend=nothing, alg::Algorithm=Auto(), dims::Union{Colon, Integer}=Colon(),
    lt::L=isless, by::B=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
) where {L, B}
    backend = resolve_backend(backend, v)
    v = one_based(v)
    ord = sort_ordering(lt, by, rev, order)
    # MergeSort sorts the keys `by.(v)` carrying `v` along
    merge_eltypes = by === identity ? (eltype(v), Nothing) :
                    (Base.promote_op(by, eltype(v)), eltype(v))
    a = resolve_sort(alg, backend, v, dims, ord; merge_eltypes)
    return (; plan=sort_plan(a, v, backend, dims, by), ord)
end

# The plan of `sort!` with each algorithm
# (merge sort lists the same buffers for every length, without elements where the call does not
# need them, so that its plans infer: the swap buffer of the global passes, or the keys `by`
# computes, sorted carrying the elements along. A zero-dimensional array, whose one element needs
# no sorting, has no empty buffers of its shape, and gets none.)
function sort_plan(a::MergeSort, v, backend, dims, by::B) where {B}
    ndims(v) == 0 && return Plan(backend, a, (;))
    len = slice_layout(v, dims).len
    if by !== identity
        K = Base.promote_op(by, eltype(v))
        sizes = (; keys=scratch(K, size(v); needed=len > 1),
                 merge_by_key_scratch(a, len, size(v), K, size(v), eltype(v))...)
        return Plan(backend, a, sizes)
    end
    return Plan(backend, a, (; temp=scratch(eltype(v), size(v); needed=len > 2 * a.block_size)))
end
sort_plan(a::RadixSort, v, backend, dims, by) = radix_plan(a, v, backend, dims)
sort_plan(a::BitonicSort, v, backend, dims, by) = Plan(backend, a, (;))
sort_plan(a::CPUThreads.SampleSort, v, backend, dims, by) =
    Plan(backend, a, dims isa Colon ? sample_sort_scratch(a, eltype(v), length(v)) : (;))

# The ordering of the sorting keywords, as `Base.Order.ord` makes it but with a `by` always
# outermost, so that MergeSort can sort the keys `by.(v)` under `ord.order` instead
function sort_ordering(lt::L, by::B, rev, order) where {L, B}
    ord = Base.Order.ord(lt, identity, rev, order)
    return by === identity ? ord : Base.Order.By(by, ord)
end

function sort_run!(a::MergeSort, v, backend, dims, ord, by::B, bufs) where {B}
    merge_sort!(v, backend, bufs, ord, by; block_size=a.block_size, dims)
end

function sort_run!(a::RadixSort, v, backend, dims, ord, by, bufs)
    radix_sort!(v, backend, bufs; rev=ord === Base.Order.Reverse,
                block_size=a.block_size, items_per_thread=a.items_per_thread,
                chunk_size=a.chunk_size, dims)
end

function sort_run!(a::BitonicSort, v, backend, dims, ord, by, bufs)
    bitonic_sort!(v, backend, ord; block_size=a.block_size, items_per_thread=a.items_per_thread,
                  dims)
end

function sort_run!(a::CPUThreads.SampleSort, v, backend, dims, ord, by, bufs)
    if dims isa Colon
        # `vec`: the local sorts use `Base.sort!`, which needs `dims` for other arrays
        sample_sort!(vec(v), ord; max_tasks=a.max_tasks, min_elems=a.min_elems, temp=bufs.temp)
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
    lt::L=isless,
    by::B=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    workspace=nothing,
) where {L, B}
    check_sortperm_shape(ix, v, dims)
    sortperm_unchecked!(ix, v; backend, alg, dims, lt, by, rev, order, workspace)
end

plan(::typeof(sortperm!), ix::AbstractArray, v::AbstractArray; dims=Colon(), kwargs...) =
    (check_sortperm_shape(ix, v, dims); sortperm_setup(ix, v; dims, kwargs...).plan)

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
                             dims::Union{Colon, Integer}=Colon(), lt::L=isless, by::B=identity,
                             rev::Union{Nothing, Bool}=nothing,
                             order::Base.Order.Ordering=Base.Order.Forward,
                             workspace=nothing) where {L, B}
    s = sortperm_setup(ix, v; backend, alg, dims, lt, by, rev, order)
    p = s.plan
    bufs = scratch_buffers(p, workspace, ix, v)
    ix_one_based = one_based(ix)
    sortperm_run!(p.alg, ix_one_based, one_based(v), p.backend, dims, s.ord, by, bufs)
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
    lt::L=isless, by::B=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
) where {L, B}
    backend = resolve_backend(backend, ix, v)
    ix, v = one_based(ix), one_based(v)
    ord = sort_ordering(lt, by, rev, order)
    merge_eltypes = (Base.promote_op(by, eltype(v)), eltype(ix))
    a = resolve_sort(alg, backend, v, dims, ord; perm=true, merge_eltypes)
    return (; plan=Plan(backend, a, sortperm_scratch(a, ix, v, dims, by)), ord)
end

# The scratch of `sortperm!` with each algorithm
function sortperm_scratch(a::MergeSort, ix, v, dims, by::B) where {B}
    (ndims(ix) == 0 || ndims(v) == 0) && return (;)
    len = slice_layout(v, dims).len
    merges = len > 2 * a.block_size
    # The keys `by.(v)` are sorted in a copy, carrying the indices along (`merge_by_key_scratch`).
    # The low-memory path sorts the indices alone: it lists the same buffers, without keys, so that
    # the plan infers whatever `lowmem` is, except for keys of a non-bits type (a union), which
    # some backends cannot allocate even empty.
    K = Base.promote_op(by, eltype(v))
    temp_values = scratch(eltype(ix), size(ix); needed=merges)
    a.lowmem && !isbitstype(K) && return (; temp_values)
    return (; keys=scratch(K, size(v); needed=!a.lowmem && len > 1),
            temp_keys=scratch(K, size(v); needed=!a.lowmem && merges), temp_values)
end
sortperm_scratch(a::CPUThreads.SampleSort, ix, v, dims, by) =
    dims isa Colon ? sample_sort_scratch(a, eltype(ix), length(ix)) : (;)

function sortperm_run!(a::MergeSort, ix, v, backend, dims, ord, by::B, bufs) where {B}
    if a.lowmem
        merge_sortperm_lowmem!(ix, v, backend, bufs, ord; block_size=a.block_size, dims)
    else
        merge_sortperm!(ix, v, backend, bufs, ord, by; block_size=a.block_size, dims)
    end
end

function sortperm_run!(a::CPUThreads.SampleSort, ix, v, backend, dims, ord, by, bufs)
    if dims isa Colon
        sample_sortperm!(vec(ix), vec(v), ord; max_tasks=a.max_tasks, min_elems=a.min_elems,
                         temp=bufs.temp)
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
    sortperm_setup(IndexShape(v), v; kwargs...).plan

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
    lt::L=isless,
    by::B=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    workspace=nothing,
) where {L, B}
    s = sort_by_key_setup(keys, values; backend, alg, dims, lt, by, rev, order)
    p = s.plan
    sort_by_key_run!(p.alg, one_based(keys), one_based(values), p.backend, dims, s.ord,
                     scratch_buffers(p, workspace, keys, values))
    return keys, values
end

plan(::typeof(sort_by_key!), keys::AbstractArray, values::AbstractArray; kwargs...) =
    sort_by_key_setup(keys, values; kwargs...).plan

function sort_by_key_setup(
    keys, values; backend=nothing, alg::Algorithm=Auto(), dims::Union{Colon, Integer}=Colon(),
    lt::L=isless, by::B=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
) where {L, B}
    backend = resolve_backend(backend, keys, values)
    ord = sort_ordering(lt, by, rev, order)
    if dims isa Colon
        length(keys) == length(values) || throw(ArgumentError(
            "keys and values must have the same length, $(length(keys)) != $(length(values))"))
    else
        axes(keys) == axes(values) || throw(ArgumentError(
            "keys and values must have the same axes, $(axes(keys)) != $(axes(values))"))
    end
    keys, values = one_based(keys), one_based(values)
    a = resolve_sort(alg, backend, keys, dims, ord; pairs=true,
                     merge_eltypes=(eltype(keys), eltype(values)))
    return (; plan=Plan(backend, a, sort_by_key_scratch(a, keys, values, dims)), ord)
end

function sort_by_key_scratch(a::MergeSort, keys, values, dims)
    (ndims(keys) == 0 || ndims(values) == 0) && return (;)
    return merge_by_key_scratch(a, slice_layout(keys, dims).len, size(keys), eltype(keys),
                                size(values), eltype(values))
end
function sort_by_key_scratch(a::CPUThreads.SampleSort, keys, values, dims)
    # The sorting permutation (sorted by `sample_sortperm!` for `dims=:`), and the old keys and
    # values to gather from
    perm = dims isa Colon ? (; perm=sample_sort_scratch(a, Int, length(keys))) : (;)
    return (; ix=scratch(Int, size(keys)), perm..., keys=scratch(eltype(keys), size(keys)),
            values=scratch(eltype(values), size(values)))
end

function sort_by_key_run!(a::MergeSort, keys, values, backend, dims, ord, bufs)
    merge_sort_by_key!(keys, values, backend, bufs, ord; block_size=a.block_size, dims)
end

function sort_by_key_run!(a::CPUThreads.SampleSort, keys, values, backend, dims, ord, bufs)
    sample_sort_by_key!(keys, values, ord, dims, bufs; max_tasks=a.max_tasks,
                        min_elems=a.min_elems)
end
