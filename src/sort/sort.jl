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
    RadixSort()

Use GPU radix sort for `sort!` and `sort`. A stable LSD radix sort (8-bit digits) supporting
32- and 64-bit integers and floats; other element types or custom `lt`/`by` fall back to
[`merge_sort!`](@ref). It is typically faster than [`MergeSort`](@ref) for these numeric types,
and skips passes automatically for small-range or structured data. Defaults to `block_size=512`.
This algorithm does not support `sortperm!`.
"""
struct RadixSort <: SortAlgorithm end

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

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Sorts the array `v` in-place using the specified backend. The `lt`, `by`, `rev`, and `order`
arguments are the same as for `Base.sort`.

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
GPU settings: use `block_size` threads per block to sort the array. When `block_size` is left as
`nothing` (the default), each GPU algorithm picks its own tuned value (256 for merge sort, 512 for
radix sort); pass an explicit `block_size` to override.

## Algorithm choice
By default, `sort!` uses [`sample_sort!`](@ref) on CPU backends and [`merge_sort!`](@ref) on GPU
backends. Pass `alg=SampleSort()` for the CPU path, `alg=MergeSort()` for the GPU merge-sort path,
or `alg=RadixSort()` to opt into GPU radix sorting. `RadixSort()` supports 32-bit and 64-bit
integers and floats; unsupported element types or custom `lt`/`by` settings fall back to
[`merge_sort!`](@ref).

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

    # GPU settings; nothing => each GPU algorithm picks its own tuned default
    block_size::Union{Nothing, Int}=nothing,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
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
            # Radix benefits from larger blocks: fewer per-block histograms shrink
            # the global histogram array, making the prefix-sum (accumulate!) cheaper.
            _radix_sort!(
                v, backend;
                lt, by, rev, order,
                block_size=isnothing(block_size) ? 512 : block_size,
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

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Save into `ix` the index permutation of `v` such that `v[ix]` is sorted. The `lt`, `by`, `rev`, and
`order` arguments are the same as for `Base.sortperm`. The same algorithms are used as for
[`sort!`](@ref) with custom by-index comparators.

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

    # GPU settings; nothing => merge sort's tuned default (sortperm is merge-only)
    block_size::Union{Nothing, Int}=nothing,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
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
