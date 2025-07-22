include("utils.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")


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

        # GPU settings
        block_size::Int=256,

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
GPU settings: use `block_size` threads per block to sort the array. A parallel [`merge_sort!`](@ref)
is used.

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

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if use_gpu_algorithm(backend, prefer_threads)
        merge_sort!(
            v, backend;
            lt, by, rev, order,
            block_size,
            temp,
        )
    else
        sample_sort!(
            v;
            lt, by, rev, order,
            max_tasks, min_elems,
            temp,
        )
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

        # GPU settings
        block_size::Int=256,

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

        # GPU settings
        block_size::Int=256,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Save into `ix` the index permutation of `v` such that `v[ix]` is sorted. The `lt`, `by`, `rev`, and
`order` arguments are the same as for `Base.sortperm`. The same algorithms are used as for
[`sort!`](@ref) with custom by-index comparators.
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

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if use_gpu_algorithm(backend, prefer_threads)
        merge_sortperm_lowmem!(
            ix, v, backend;
            lt, by, rev, order,
            block_size, temp,
        )
    else
        sample_sortperm!(
            ix, v;
            lt, by, rev, order,
            max_tasks,
            min_elems,
            temp,
        )
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

        # GPU settings
        block_size::Int=256,

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
