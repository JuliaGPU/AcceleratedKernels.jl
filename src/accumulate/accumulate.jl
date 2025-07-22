# Available accumulation algorithms
abstract type AccumulateAlgorithm end
struct DecoupledLookback <: AccumulateAlgorithm end
struct ScanPrefixes <: AccumulateAlgorithm end


# Helpers
# Check the given dst is compatible with src and init
function _accumulate_check_types(dst, src, init)
    eltype(dst) === eltype(src) && return
    eltype(dst) === typeof(init) && return
    eltype(dst) === promote_type(eltype(src), typeof(init)) && return

    throw(ArgumentError(
        """
        destination array type `$(eltype(dst))` (temp) is incompatible with source array type
        `$(eltype(src))` and initial value type `$(typeof(init))`; eltype(dst) must be either
        like eltype(src) or typeof(init) or promote_type(eltype(src), typeof(init)).
        """
    ))
end


# Implementations, then interfaces
include("accumulate_1d_cpu.jl")
include("accumulate_1d_gpu.jl")
include("accumulate_nd.jl")


"""
    accumulate!(
        op, v::AbstractArray, backend::Backend=get_backend(v);
        init,
        neutral=neutral_element(op, eltype(v)),
        dims::Union{Nothing, Int}=nothing,
        inclusive::Bool=true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=2,

        # Algorithm choice
        alg::AccumulateAlgorithm=DecoupledLookback(),

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

    accumulate!(
        op, dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(v);
        init,
        neutral=neutral_element(op, eltype(dst)),
        dims::Union{Nothing, Int}=nothing,
        inclusive::Bool=true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=2,

        # Algorithm choice
        alg::AccumulateAlgorithm=DecoupledLookback(),

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Compute accumulated running totals along a sequence by applying a binary operator to all elements
up to the current one; often used in GPU programming as a first step in finding / extracting
subsets of data.

**Other names**: prefix sum, `thrust::scan`, cumulative sum; inclusive (or exclusive) if the first
element is included in the accumulation (or not).

For compatibility with the `Base.accumulate!` function, we provide the two-array interface too, but
we do not need the constraint of `dst` and `src` being different; to minimise memory use, we
recommend using the single-array interface (the first one above).

## CPU
Use at most `max_tasks` threads with at least `min_elems` elements per task.

Note that accumulation is typically a memory-bound operation, so multithreaded accumulation only
becomes faster if it is a more compute-heavy operation to hide memory latency - that includes:
- Accumulating more complex types, e.g. accumulation of tuples / structs / strings.
- More complex operators, e.g. `op=custom_complex_function`.

## GPU
For the 1D case (`dims=nothing`), the `alg` can be one of the following:
- `DecoupledLookback()`: the default algorithm, using opportunistic lookback to reuse earlier
  blocks' results; requires device-level memory consistency guarantees, which Apple Metal does not
  provide.
- `ScanPrefixes()`: a simpler algorithm that scans the prefixes of each block, with no lookback; it
  has similar performance as `DecoupledLookback()` for large block sizes, and small to medium arrays,
  but poorer scaling for many blocks; there is no performance degradation below `block_size^2`
  elements.

A different, unique algorithm is used for the multi-dimensional case (`dims` is an integer).

The `block_size` should be a power of 2 and greater than 0.

The temporaries are only used for the 1D case (`dims=nothing`): `temp` stores per-block aggregates;
`temp_flags` is only used for the `DecoupledLookback()` algorithm for flagging if blocks are ready;
they should both have at least `(length(v) + 2 * block_size - 1) ÷ (2 * block_size)` elements; also,
`eltype(v) === eltype(temp)` is required; the elements in `temp_flags` can be any integers, but
`Int8` is used by default to reduce memory usage.

# Platform-Specific Notes
On Metal, the `alg=ScanPrefixes()` algorithm is used by default, as Apple Metal GPUs do not have
strong enough memory consistency guarantees for the `DecoupledLookback()` algorithm - which
produces incorrect results about 0.38% of the time (the beauty of parallel algorithms, ey). Also,
`block_size=1024` is used here by default to reduce the number of coupled lookbacks.

# Examples
Example computing an inclusive prefix sum (the typical GPU "scan"):
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneAPI.ones(Int32, 100_000)
AK.accumulate!(+, v, init=0)

# Use a different algorithm
AK.accumulate!(+, v, alg=AK.ScanPrefixes())
```
"""
function accumulate!(
    op, v::AbstractArray, backend::Backend=get_backend(v);
    init,
    kwargs...
)
    _accumulate_impl!(
        op, v, backend;
        init,
        kwargs...
    )
end


function accumulate!(
    op, dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(dst);
    init,
    kwargs...
)
    copyto!(dst, src)
    _accumulate_impl!(
        op, dst, backend;
        init,
        kwargs...
    )
end


function _accumulate_impl!(
    op, v::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(v)),
    dims::Union{Nothing, Int}=nothing,
    inclusive::Bool=true,

    # FIXME: Switch back to `DecoupledLookback()` as the default algorithm
    #         once https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/44 is merged.
    alg::AccumulateAlgorithm=ScanPrefixes(),

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=2,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    if isnothing(dims)
        return if use_gpu_algorithm(backend, prefer_threads)
            accumulate_1d_gpu!(
                op, v, backend, alg;
                init, neutral, inclusive,
                max_tasks, min_elems,
                block_size, temp, temp_flags,
            )
        else
            accumulate_1d_cpu!(
                op, v, backend, alg;
                init, neutral, inclusive,
                max_tasks, min_elems,
                block_size, temp, temp_flags,
            )
        end
    else
        return accumulate_nd!(
            op, v, backend;
            init, neutral, dims, inclusive,
            max_tasks, min_elems, prefer_threads,
            block_size,
        )
    end
end


"""
    accumulate(
        op, v::AbstractArray, backend::Backend=get_backend(v);
        init,
        neutral=neutral_element(op, eltype(v)),
        dims::Union{Nothing, Int}=nothing,
        inclusive::Bool=true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=2,

        # Algorithm choice
        alg::AccumulateAlgorithm=DecoupledLookback(),

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Out-of-place version of [`accumulate!`](@ref).
"""
function accumulate(
    op, v::AbstractArray, backend::Backend=get_backend(v);
    init,
    kwargs...
)
    dst_type = Base.promote_op(op, eltype(v), typeof(init))
    vcopy = similar(v, dst_type)
    copyto!(vcopy, v)
    accumulate!(
        op, vcopy, backend;
        init,
        kwargs...
    )
    vcopy
end

