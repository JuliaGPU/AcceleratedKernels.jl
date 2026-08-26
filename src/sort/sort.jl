include("utils.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")
include("radix_sort.jl")


# Available sorting algorithms
abstract type SortAlgorithm end

"""
    MergeSort(; lowmem=false)

Use GPU merge sort for `sort!` and `sort`. For `sortperm!`, `lowmem=true` selects the
lower-memory permutation path.
"""
Base.@kwdef struct MergeSort <: SortAlgorithm
    lowmem::Bool = false
end

"""
    RadixSort(; block_size=nothing, items_per_thread=nothing)

Use GPU radix sort for `sort!` and `sort`. Supports `UInt32`, `Int32`, `Float32`, `UInt64`,
`Int64`, and `Float64` with forward or reverse ordering. This algorithm does not support
`sortperm!`.
"""
Base.@kwdef struct RadixSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end

_radix_defaults(::Backend) = (block_size=256, items_per_thread=2)

"""
    SampleSort()

Use CPU sample sort for `sort!`, `sort`, `sortperm!`, and `sortperm`.
"""
struct SampleSort <: SortAlgorithm end


# All other algorithms have the same naming convention as Julia Base ones; provide similar
# interface here too.


"""
    sort!(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # Algorithm choice
        alg::Union{Nothing, SortAlgorithm}=nothing,

        # Sort each slice along this dimension; `:` sorts the whole array flat
        dims::Union{Colon, Integer}=Colon(),

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Sorts the array `v` in-place using the specified backend. The `lt`, `by`, `rev`, and `order`
arguments are the same as for `Base.sort`.

With the default `dims=:` the whole array is sorted as one flat vector. Pass an integer `dims` to
sort each 1D slice along that dimension independently, matching `Base.sort(A; dims)`. The `dims`
form always uses a comparison sort and so ignores `alg`.

## CPU
CPU settings: use at most `max_tasks` threads to sort the array such that at least `min_elems`
elements are sorted by each thread. A parallel [`sample_sort!`](@ref) is used, processing
independent slices of the array and deferring to `Base.sort!` for the final local sorts.

Note that the Base Julia `sort!` is mainly memory-bound, so multithreaded sorting only becomes
faster if it is a more compute-heavy operation to hide memory latency - that includes:
- Sorting more complex types, e.g. lexicographic sorting of tuples / structs / strings.
- More complex comparators, e.g. `by=custom_complex_function` or `lt=custom_lt_function`.
- Less cache-predictable data movement, e.g. `sortperm`.

## GPU
GPU settings: `block_size` sets the number of threads per block. For `RadixSort`, fields on the
algorithm take precedence over this keyword, then backend defaults. `items_per_thread` is set on
`RadixSort` and defaults to 2.

## Algorithm choice
By default, `sort!` uses [`sample_sort!`](@ref) on CPU backends and [`merge_sort!`](@ref) on GPU
backends. Pass `alg=SampleSort()` for the CPU path, `alg=MergeSort()` for the GPU merge-sort path,
or `alg=RadixSort()` to opt into GPU radix sorting. `RadixSort()` supports 32-bit and 64-bit
integers and floats with default `lt`/`by`.

For both CPU and GPU backends, the `temp` argument can be used to reuse a temporary buffer of the
same size as `v` to store the sorted output.

# Examples
Simple parallel CPU sort using all available threads (as given by `julia --threads N`):
```julia
import AcceleratedKernels as AK
v = rand(1000)
AK.sort!(v)
```

Parallel GPU sorting, passing a temporary buffer to avoid allocating a new one:
```julia
using oneAPI
import AcceleratedKernels as AK
v = oneArray(rand(1000))
temp = similar(v)
AK.sort!(v, temp=temp)
```
"""
function sort!(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    _sort_impl!(
        v, backend;
        kwargs...
    )
end


function _sort_impl!(
    v::AbstractArray, backend::Backend;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,

    max_tasks=Threads.nthreads(),
    min_elems=1,
    prefer_threads::Bool=true,

    alg::Union{Nothing, SortAlgorithm}=nothing,

    # Sort each 1D slice along this dimension; `:` sorts the whole array as one flat vector
    dims::Union{Colon, Integer}=Colon(),

    # GPU settings; nothing => each GPU algorithm picks its own tuned default
    block_size::Union{Nothing, Int}=nothing,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if !(dims isa Colon)
        return _sort_dims!(
            v, backend, Int(dims);
            lt, by, rev, order,
            max_tasks, min_elems, prefer_threads,
            block_size,
        )
    end

    if use_gpu_algorithm(backend, prefer_threads)
        alg = isnothing(alg) ? MergeSort() : alg
        if alg isa MergeSort
            merge_sort!(
                v, backend;
                lt, by, rev, order,
                block_size=isnothing(block_size) ? 256 : block_size,
                temp,
            )
        elseif alg isa RadixSort
            _rs_supported(eltype(v)) || throw(ArgumentError("RadixSort is not supported for eltype \"$(eltype(v))\""))
            ordering = Base.Order.ord(lt, by, rev, order)
            ordering === Base.Order.Forward || ordering === Base.Order.Reverse ||
                throw(ArgumentError("RadixSort only supports forward or reverse ordering"))
            defaults = _radix_defaults(backend)
            radix_block_size = isnothing(alg.block_size) ?
                               (isnothing(block_size) ? defaults.block_size : block_size) : alg.block_size
            radix_items = isnothing(alg.items_per_thread) ?
                          defaults.items_per_thread : alg.items_per_thread
            _radix_sort!(
                v, backend;
                descending=ordering === Base.Order.Reverse,
                block_size=radix_block_size,
                items_per_thread=radix_items,
                temp,
            )
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sort! on GPU backends"))
        end
    else
        alg = isnothing(alg) ? SampleSort() : alg
        if alg isa SampleSort
            sample_sort!(
                v;
                lt, by, rev, order,
                max_tasks, min_elems,
                temp,
            )
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sort! on CPU backends"))
        end
    end
end


# Sort each slice along `dim` on its own, like Base.sort(A; dims).
# We have no batched sort kernel, so tag every element with its slice, sort the whole array
# once by (slice, value), then scatter each element back to its place. Works on any backend.
function _sort_dims!(
    v::AbstractArray{T, N}, backend::Backend, dim::Int;
    lt, by, rev, order,
    max_tasks, min_elems, prefer_threads,
    block_size,
) where {T, N}
    1 <= dim <= N || throw(ArgumentError("dimension $dim is not 1 ≤ dims ≤ $N"))
    slice_len = size(v, dim)
    (length(v) <= 1 || slice_len <= 1) && return v     # every slice is a singleton

    len = length(v)
    bs = isnothing(block_size) ? 256 : block_size

    # a slice is picked by the other dims, so collapse `dim` to 1
    other_size = ntuple(d -> d == dim ? 1 : size(v, d), N)
    slice_lin = LinearIndices(other_size)
    slice_car = CartesianIndices(other_size)
    elem_car = CartesianIndices(v)

    keys = similar(v, Tuple{Int, T}, len)
    foreachindex(v, backend; max_tasks, min_elems, prefer_threads, block_size=bs) do i
        ci = elem_car[i]
        proj = ntuple(d -> d == dim ? 1 : ci[d], N)
        sid = slice_lin[CartesianIndex(proj)] - 1
        @inbounds keys[i] = (sid, v[i])
    end

    # slices in order, values within a slice by the user's ordering
    o = Base.Order.ord(lt, by, rev, order)
    comp = (a, b) -> a[1] != b[1] ? a[1] < b[1] : Base.Order.lt(o, a[2], b[2])
    _sort_impl!(keys, backend; lt=comp, max_tasks, min_elems, prefer_threads, block_size)

    # sorted keys are grouped by slice, so slot k lands at row (k-1)%slice_len of its slice
    foreachindex(keys, backend; max_tasks, min_elems, prefer_threads, block_size=bs) do k
        s = (k - 1) ÷ slice_len
        r = (k - 1) % slice_len
        oc = slice_car[s + 1]
        dst = ntuple(d -> d == dim ? r + 1 : oc[d], N)
        @inbounds v[CartesianIndex(dst)] = keys[k][2]
    end

    v
end


"""
    sort(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # Algorithm choice
        alg::Union{Nothing, SortAlgorithm}=nothing,

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Out-of-place sort, same settings as [`sort!`](@ref).
"""
function sort(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    vcopy = copy(v)
    sort!(
        vcopy, backend;
        kwargs...
    )
end


"""
    sortperm!(
        ix::AbstractArray,
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # Algorithm choice
        alg::Union{Nothing, SortAlgorithm}=nothing,

        # Permute each slice along this dimension; `:` permutes the whole array flat
        dims::Union{Colon, Integer}=Colon(),

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Save into `ix` the index permutation of `v` such that `v[ix]` is sorted. The `lt`, `by`, `rev`, and
`order` arguments are the same as for `Base.sortperm`. The same algorithms are used as for
[`sort!`](@ref) with custom by-index comparators.

With the default `dims=:` the whole array is permuted as one flat vector. Pass an integer `dims` to
permute each 1D slice along that dimension independently, matching `Base.sortperm(A; dims)`; then
`ix` holds global linear indices and must have the same axes as `v`. The `dims` form always uses a
comparison sort and so ignores `alg`.

## Algorithm choice
By default, `sortperm!` uses [`sample_sortperm!`](@ref) on CPU backends and [`merge_sortperm!`](@ref)
on GPU backends. Pass `alg=MergeSort(lowmem=true)` to use the lower-memory GPU permutation path.
`RadixSort()` does not provide a permutation path.
"""
function sortperm!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);
    kwargs...
)
    _sortperm_impl!(
        ix, v, backend;
        kwargs...
    )
end


function _sortperm_impl!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,

    max_tasks=Threads.nthreads(),
    min_elems=1,
    prefer_threads::Bool=true,

    alg::Union{Nothing, SortAlgorithm}=nothing,

    # Permute each slice along this dimension; `:` permutes the whole array as one flat vector
    dims::Union{Colon, Integer}=Colon(),

    # GPU settings; nothing => merge sort's tuned default (sortperm is merge-only)
    block_size::Union{Nothing, Int}=nothing,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if !(dims isa Colon)
        return _sortperm_dims!(
            ix, v, backend, Int(dims);
            lt, by, rev, order,
            max_tasks, min_elems, prefer_threads,
            block_size,
        )
    end

    if use_gpu_algorithm(backend, prefer_threads)
        alg = isnothing(alg) ? MergeSort() : alg
        bs = isnothing(block_size) ? 256 : block_size
        if alg isa MergeSort
            if alg.lowmem
                merge_sortperm_lowmem!(
                    ix, v, backend;
                    lt, by, rev, order,
                    block_size=bs,
                    temp,
                )
            else
                # merge_sortperm! copies keys alongside indices in shared memory so comparisons
                # never touch global memory during the binary-search step.
                # merge_sortperm_lowmem! avoids the key copy but its comparator does two global
                # loads per comparison, making it O(n log²n) in global traffic at large n.
                merge_sortperm!(
                    ix, v, backend;
                    lt, by, rev, order,
                    block_size=bs,
                    temp_ix=temp,   # old `temp` was the index buffer; maps directly to temp_ix
                )
            end
        elseif alg isa RadixSort
            throw(ArgumentError("RadixSort does not support sortperm"))
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sortperm! on GPU backends"))
        end
    else
        alg = isnothing(alg) ? SampleSort() : alg
        if alg isa SampleSort
            sample_sortperm!(
                ix, v;
                lt, by, rev, order,
                max_tasks,
                min_elems,
                temp,
            )
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sortperm! on CPU backends"))
        end
    end
end


# Same idea as _sort_dims!, but produce the permutation instead of sorting in place.
# We tag with (slice, value, index), sort, and write the original index into `ix`. The index
# also breaks ties, which keeps the permutation stable like Base.sortperm(A; dims).
function _sortperm_dims!(
    ix::AbstractArray, v::AbstractArray{T, N}, backend::Backend, dim::Int;
    lt, by, rev, order,
    max_tasks, min_elems, prefer_threads,
    block_size,
) where {T, N}
    1 <= dim <= N || throw(ArgumentError("dimension $dim is not 1 ≤ dims ≤ $N"))
    axes(ix) == axes(v) || throw(ArgumentError("index array must have the same axes as the input"))
    slice_len = size(v, dim)
    len = length(v)
    bs = isnothing(block_size) ? 256 : block_size

    if len <= 1 || slice_len <= 1                      # each slice holds one element
        foreachindex(v, backend; max_tasks, min_elems, prefer_threads, block_size=bs) do i
            @inbounds ix[i] = i
        end
        return ix
    end

    # a slice is picked by the other dims, so collapse `dim` to 1
    other_size = ntuple(d -> d == dim ? 1 : size(v, d), N)
    slice_lin = LinearIndices(other_size)
    slice_car = CartesianIndices(other_size)
    elem_car = CartesianIndices(v)

    keys = similar(v, Tuple{Int, T, Int}, len)
    foreachindex(v, backend; max_tasks, min_elems, prefer_threads, block_size=bs) do i
        ci = elem_car[i]
        proj = ntuple(d -> d == dim ? 1 : ci[d], N)
        sid = slice_lin[CartesianIndex(proj)] - 1
        @inbounds keys[i] = (sid, v[i], i)
    end

    # slices in order, then values, then index to break ties so the result stays stable
    o = Base.Order.ord(lt, by, rev, order)
    comp = (a, b) -> begin
        a[1] != b[1] && return a[1] < b[1]
        Base.Order.lt(o, a[2], b[2]) && return true
        Base.Order.lt(o, b[2], a[2]) && return false
        return a[3] < b[3]
    end
    _sort_impl!(keys, backend; lt=comp, max_tasks, min_elems, prefer_threads, block_size)

    # sorted keys are grouped by slice, so slot k lands at row (k-1)%slice_len of its slice
    foreachindex(keys, backend; max_tasks, min_elems, prefer_threads, block_size=bs) do k
        s = (k - 1) ÷ slice_len
        r = (k - 1) % slice_len
        oc = slice_car[s + 1]
        dst = ntuple(d -> d == dim ? r + 1 : oc[d], N)
        @inbounds ix[CartesianIndex(dst)] = keys[k][3]
    end

    ix
end


"""
    sortperm(
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # Algorithm choice
        alg::Union{Nothing, SortAlgorithm}=nothing,

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Out-of-place sortperm, same settings as [`sortperm!`](@ref).
"""
function sortperm(
    v::AbstractArray,
    backend::Backend=get_backend(v);
    kwargs...
)
    ix = similar(v, Int)
    sortperm!(
        ix, v, backend;
        kwargs...
    )
end
