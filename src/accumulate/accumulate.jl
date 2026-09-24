include("tuning.jl")


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
    accumulate!(op, v::AbstractArray; kwargs...) -> v
    accumulate!(op, dst::AbstractArray, src::AbstractArray; kwargs...) -> dst

    accumulate!(
        op, dst, src;
        backend=nothing,
        init,
        neutral=neutral_element(op, eltype(dst)),
        dims::Union{Nothing, Integer}=nothing,
        inclusive::Bool=true,
        alg::Algorithm=Auto(),
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Compute accumulated running totals along a sequence by applying a binary operator to all elements
up to the current one; often used in GPU programming as a first step in finding / extracting
subsets of data. The first form scans `v` in place; the second writes the scan of `src` to `dst`
(which may be `src` itself).

**Other names**: prefix sum, `thrust::scan`, cumulative sum; inclusive (or exclusive) if the first
element is included in the accumulation (or not).

The operator `op` must be associative, as elements are combined in parallel; it does not need to
be commutative, as every combination keeps the elements in order (e.g. matrix products are fine).

With `dims=nothing` the array is scanned in linear order, whatever its number of dimensions; with
an integer `dims`, each slice along that dimension is scanned independently.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host, and on GPUs [`ScanPrefixes`](@ref) for
whole arrays and [`SliceScan`](@ref) along `dims`, with the device's settings.
[`DecoupledLookback`](@ref) is available on backends that support it. `backend` is derived from
`dst` and `src`.

On the host, accumulation is typically a memory-bound operation, so multithreaded accumulation
only becomes faster for more compute-heavy operations that hide memory latency, e.g. accumulating
tuples or structs, or expensive operators.

The temporaries only apply to whole-array GPU scans: `temp` stores per-block aggregates, with
`eltype(temp) === eltype(dst)`; `temp_flags` stores `DecoupledLookback`'s block flags (any
integer type). Both need at least `cld(length(dst), block_size * items_per_thread)` elements
of the resolved algorithm. Multi-block exclusive scans with `DecoupledLookback` use two epilogue
kernels to shift the result in place; they reuse `temp` for tile-boundary values and do not
allocate a full-array copy.

# Examples
Example computing an inclusive prefix sum (the typical GPU "scan"):
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneAPI.ones(Int32, 100_000)
AK.accumulate!(+, v; init=0)

# Choose the algorithm and its settings
AK.accumulate!(+, v; init=0, alg=AK.ScanPrefixes(block_size=512))
```
"""
function accumulate!(op, v::AbstractArray; backend::Union{Nothing, Backend}=nothing, kwargs...)
    _accumulate_impl!(op, v, v, _resolve_backend(backend, v); kwargs...)
end

function accumulate!(
    op, dst::AbstractArray, src::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    kwargs...
)
    _accumulate_impl!(op, dst, src, _resolve_backend(backend, dst, src); kwargs...)
end


# Scan `src` into `v` (which may be `src`), after checking the algorithm
function _accumulate_impl!(
    op, v::AbstractArray, src::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(v)),
    dims::Union{Nothing, Integer}=nothing,
    inclusive::Bool=true,
    alg::Algorithm=Auto(),
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    dims isa Integer && dims < 1 &&
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    a = _resolve_scan(alg, backend, eltype(v), dims)
    if dims !== nothing && (temp !== nothing || temp_flags !== nothing)
        throw(ArgumentError(
            "`temp` and `temp_flags` only apply to whole-array scans (`dims=nothing`)"))
    end
    v === src || copyto!(v, src)
    if dims === nothing
        if a isa CPUThreads.Partitioned
            accumulate_1d_cpu!(op, v, backend, a; init, neutral, inclusive)
        else
            accumulate_1d_gpu!(op, v, backend, a; init, neutral, inclusive, temp, temp_flags)
        end
    else
        accumulate_nd!(op, v, backend, a; init, neutral, dims=Int(dims), inclusive)
    end
    return v
end


"""
    accumulate(op, v::AbstractArray; init, kwargs...)

Out-of-place version of [`accumulate!`](@ref), with the same keywords; the result's element type
is `Base.promote_op(op, eltype(v), typeof(init))`.
"""
function accumulate(op, v::AbstractArray; init, kwargs...)
    dst_type = Base.promote_op(op, eltype(v), typeof(init))
    vcopy = similar(v, dst_type)
    copyto!(vcopy, v)
    accumulate!(op, vcopy; init, kwargs...)
end
