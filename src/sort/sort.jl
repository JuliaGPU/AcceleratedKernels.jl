include("utils.jl")
include("slices.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")
include("radix_sort.jl")
include("bitonic_sort.jl")


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
    BitonicSort(; block_size=nothing)

Use a GPU bitonic sort for `sort!` and `sort`. Supports `UInt32`, `Int32`, `Float32`, `UInt64`,
`Int64`, and `Float64` with forward or reverse ordering. This algorithm does not support
`sortperm!`.

A bitonic sorting network has no data-dependent control flow, so every comparator runs
unconditionally — a good fit for GPUs. For an input that fits a single workgroup's shared memory
the whole sort runs in one kernel with no global-memory round-trips, which is faster than the other
GPU algorithms for small arrays. Above that size the network spills to global memory and its
`O(n·log²n)` comparisons cost more than the `O(n)`-pass [`RadixSort`](@ref) or `MergeSort`, so it
grows slower with `n`:

- Small arrays (up to roughly a single workgroup, a few thousand elements): fastest of the three.
- Medium arrays: comparable to `MergeSort`.
- Large arrays: slower than `RadixSort` (measured ~1.1–1.8× on an RTX 5080 and RX 9060 XT), because
  the number of global compare-exchange passes grows with `log²n`.

Non-power-of-two lengths are padded up internally; `block_size` sets the threads per workgroup.
"""
Base.@kwdef struct BitonicSort <: SortAlgorithm
    block_size::Union{Nothing, Int} = nothing
end

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

        # Sort each 1D slice along this dimension; `:` sorts the whole array as one vector
        dims::Union{Colon, Integer}=Colon(),

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Sorts the array `v` in-place using the specified backend. The `lt`, `by`, `rev`, and `order`
arguments are the same as for `Base.sort`.

By default (`dims=:`) the whole array is sorted as one vector, whatever its shape. Pass an integer
`dims` to sort each 1D slice along that dimension independently, like `Base.sort!(A; dims)`. On GPU
backends the slices are sorted with the same merge sort as the whole-array case, each slice its own
sequence of merges; `RadixSort` does not support `dims`. On CPU backends each slice is sorted with
`Base.sort!` (also with `alg=SampleSort()`, and without using `temp`), slices spread over the tasks.

## CPU
CPU settings: use at most `max_tasks` threads to sort the array such that at least `min_elems`
elements are sorted by each thread. A parallel sample sort is used, processing
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
By default, `sort!` uses sample sort on CPU backends and merge sort on GPU
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

    # Sort each 1D slice along this dimension; `:` sorts the whole array as one vector
    dims::Union{Colon, Integer}=Colon(),

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
                temp, dims,
            )
        elseif alg isa RadixSort
            dims isa Colon || throw(ArgumentError("RadixSort does not support sorting along `dims`"))
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
        elseif alg isa BitonicSort
            v isa AbstractVector ||
                throw(ArgumentError("BitonicSort only supports vectors (got a $(ndims(v))-dimensional array)"))
            _bs_supported(eltype(v)) || throw(ArgumentError("BitonicSort is not supported for eltype \"$(eltype(v))\""))
            ordering = Base.Order.ord(lt, by, rev, order)
            ordering === Base.Order.Forward || ordering === Base.Order.Reverse ||
                throw(ArgumentError("BitonicSort only supports forward or reverse ordering"))
            _bitonic_sort!(
                v, backend;
                descending=ordering === Base.Order.Reverse,
                block_size=isnothing(alg.block_size) ? block_size : alg.block_size,
            )
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sort! on GPU backends"))
        end
    else
        alg = isnothing(alg) ? SampleSort() : alg
        if !(alg isa SampleSort)
            throw(ArgumentError("$(typeof(alg)) is not supported by sort! on CPU backends"))
        elseif dims isa Colon
            sample_sort!(
                v;
                lt, by, rev, order,
                max_tasks, min_elems,
                temp,
            )
        else
            ord = Base.Order.ord(lt, by, rev, order)
            foreach_slice(v, dims; max_tasks, min_elems) do slice
                Base.sort!(slice; order=ord)
            end
        end
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

        # Sort each 1D slice along this dimension; `:` sorts the whole array as one vector
        dims::Union{Colon, Integer}=Colon(),

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

        # Permute each 1D slice along this dimension; `:` permutes the whole array as one vector
        dims::Union{Colon, Integer}=Colon(),

        # GPU settings
        block_size::Union{Nothing, Int}=nothing,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Save into `ix` the index permutation of `v` such that `v[ix]` is sorted. The `lt`, `by`, `rev`, and
`order` arguments are the same as for `Base.sortperm`. The same algorithms are used as for
[`sort!`](@ref) with custom by-index comparators.

By default (`dims=:`) the whole array is permuted as one vector. Pass an integer `dims` to permute
each 1D slice along that dimension independently, like `Base.sortperm!(ix, A; dims)`: `ix` must
then have the same axes as `v` and receives linear indices into `v`, so that `v[ix]` is sorted
along `dims`. The permutation is stable in both cases.

## Algorithm choice
By default, `sortperm!` uses sample sort on CPU backends and merge sort on GPU
backends. Pass `alg=MergeSort(lowmem=true)` to use the lower-memory GPU permutation path.
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

    # Permute each 1D slice along this dimension; `:` permutes the whole array as one vector
    dims::Union{Colon, Integer}=Colon(),

    # GPU settings; nothing => merge sort's tuned default (sortperm is merge-only)
    block_size::Union{Nothing, Int}=nothing,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if !(dims isa Colon)
        1 <= dims <= ndims(v) || throw(ArgumentError("dimension out of range"))
        axes(ix) == axes(v) ||
            throw(ArgumentError("index array must have the same axes as the input, $(axes(ix)) != $(axes(v))"))
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
                    temp, dims,
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
                    dims,
                )
            end
        elseif alg isa RadixSort
            throw(ArgumentError("RadixSort does not support sortperm"))
        else
            throw(ArgumentError("$(typeof(alg)) is not supported by sortperm! on GPU backends"))
        end
    else
        alg = isnothing(alg) ? SampleSort() : alg
        if !(alg isa SampleSort)
            throw(ArgumentError("$(typeof(alg)) is not supported by sortperm! on CPU backends"))
        elseif dims isa Colon
            sample_sortperm!(
                ix, v;
                lt, by, rev, order,
                max_tasks,
                min_elems,
                temp,
            )
        else
            # Like Base: sort the linear indices of each slice by the values they point to
            ord = Base.Order.Perm(Base.Order.ord(lt, by, rev, order), vec(v))
            copyto!(ix, LinearIndices(v))
            foreach_slice(ix, dims; max_tasks, min_elems) do slice
                Base.sort!(slice; order=ord)
            end
        end
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

        # Permute each 1D slice along this dimension; `:` permutes the whole array as one vector
        dims::Union{Colon, Integer}=Colon(),

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
