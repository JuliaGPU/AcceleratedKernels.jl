# Backend implementations
include("utilities.jl")

const MapReduceSource = Union{AbstractArray, Base.Broadcast.Broadcasted}

_mapreduce_eltype(src::AbstractArray) = eltype(src)
_mapreduce_eltype(src::Base.Broadcast.Broadcasted) =
    Base.Broadcast.combine_eltypes(identity, (src,))

function _mapreduce_backend(src::AbstractArray)
    return _mapreduce_get_backend(src)
end

function _mapreduce_backend(src::Base.Broadcast.Broadcasted)
    backend = _mapreduce_backend_from_args(src.args)
    return isnothing(backend) ? CPU_BACKEND : backend
end

function _mapreduce_get_backend(src::AbstractArray)
    try
        return get_backend(src)
    catch err
        err isa ArgumentError || rethrow()
        return CPU_BACKEND
    end
end

_mapreduce_backend_from_arg(src::AbstractArray) = _mapreduce_get_backend(src)
_mapreduce_backend_from_arg(src::Base.Broadcast.Broadcasted) = _mapreduce_backend(src)
_mapreduce_backend_from_arg(_) = nothing

function _mapreduce_backend_from_args(args::Tuple)
    backend = nothing
    for arg in args
        arg_backend = _mapreduce_backend_from_arg(arg)
        isnothing(arg_backend) && continue
        if isnothing(backend)
            backend = arg_backend
        else
            @argcheck arg_backend == backend
        end
    end
    return backend
end

function _mapreduce_check_map_axes(src::AbstractArray, srcs::AbstractArray...)
    src_axes = axes(src)
    for other in srcs
        axes(other) == src_axes || throw(DimensionMismatch("all input arrays must have the same axes"))
    end
    return nothing
end

include("mapreduce_1d_cpu.jl")
include("mapreduce_1d_gpu.jl")
include("mapreduce_nd.jl")


"""
    reduce(
        op, src::AbstractArray, backend::Backend=get_backend(src);
        init,
        neutral=neutral_element(op, eltype(src)),
        dims::Union{Nothing, Int, Tuple{Vararg{Int}}, Colon}=nothing,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op`. If `dims` is `nothing` or
`:`, reduce `src` to a scalar. If `dims` is an integer or a tuple of integers, reduce `src` along
those dimension(s). The `init` value is used as the initial value for the reduction; `neutral` is
the neutral element for the operator `op`.

The returned type is the same as `init` - to control output precision, specify `init` explicitly.

## CPU settings
Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
cases are scaling linearly with the number of threads.

Note that multithreading reductions only improves performance for cases with more compute-heavy
operations, which hide the memory latency and thread launch overhead - that includes:
- Reducing more complex types, e.g. reduction of tuples / structs / strings.
- More complex operators, e.g. `op=custom_complex_op_function`.

For non-memory-bound operations, reductions scale almost linearly with the number of threads.

## GPU settings
The `block_size` parameter controls the number of threads per block.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing` or `dims=:`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 *
block_size)` is required. For reduction along dimensions (`dims` is an integer or tuple), `temp` is
used as the destination array, and thus must have the exact dimensions required - i.e. same
dimensionwise sizes as `src`, except for the reduced dimension(s) which become 1; there are some
corner cases when one dimension is zero, check against `Base.reduce` for CPU arrays for exact
behavior.

The `switch_below` parameter controls the threshold below which the reduction is performed on the
CPU and is only used for 1D reductions (i.e. `dims=nothing`).

# Examples
Computing a sum, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsum = AK.reduce((x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsum = AK.reduce(+, m; init=zero(eltype(m)), dims=1)
mcolsum = AK.reduce(+, m; init=zero(eltype(m)), dims=2)
```
"""
function reduce(
    op, src::AbstractArray, backend::Backend=_mapreduce_backend(src);
    init,
    kwargs...
)
    _mapreduce_impl(
        identity, op, src, backend;
        init,
        kwargs...
    )
end




"""
    mapreduce(
        f, op, src::AbstractArray, backend::Backend=get_backend(src);
        init,
        neutral=neutral_element(op, eltype(src)),
        dims::Union{Nothing, Int, Tuple{Vararg{Int}}, Colon}=nothing,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

    mapreduce(f, op, A::AbstractArray, B::AbstractArray, As::AbstractArray...; init, kwargs...)

Reduce `src` along dimensions `dims` using the binary operator `op` after applying `f` elementwise.
If `dims` is `nothing` or `:`, reduce `src` to a scalar. If `dims` is an integer or a tuple of
integers, reduce `src` along those dimension(s). The `init` value is used as the initial value for
the reduction (i.e. after mapping).

The `neutral` value is the neutral element (zero) for the operator `op`, which is needed for an
efficient GPU implementation that also allows a nonzero `init`.

The returned type is the same as `init` - to control output precision, specify `init` explicitly.

Multiple input arrays are supported with the same axes. This follows `Base.mapreduce(f, op, A, B,
...)` semantics: `f` is mapped across corresponding elements of the inputs and the mapped values
are reduced without materializing the intermediate array. Mismatched axes throw
`DimensionMismatch`; singleton-expanding broadcast semantics are reserved for internal
`Broadcasted` sources used by array backends.

## CPU settings
Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
cases are scaling linearly with the number of threads.

## GPU settings
The `block_size` parameter controls the number of threads per block.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing` or `dims=:`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 *
block_size)` is required. For reduction along dimensions (`dims` is an integer or tuple), `temp` is
used as the destination array, and thus must have the exact dimensions required - i.e. same
dimensionwise sizes as `src`, except for the reduced dimension(s) which become 1; there are some
corner cases when one dimension is zero, check against `Base.reduce` for CPU arrays for exact
behavior.

The `switch_below` parameter controls the threshold below which the reduction is performed on the
CPU and is only used for 1D reductions (i.e. `dims=nothing`).

# Example
Computing a sum of squares, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsumsq = AK.mapreduce(x -> x * x, (x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums of squares in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

f(x) = x * x
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsumsq = AK.mapreduce(f, +, m; init=zero(eltype(m)), dims=1)
mcolsumsq = AK.mapreduce(f, +, m; init=zero(eltype(m)), dims=2)
```

Computing a two-input dimensional reduction:
```julia
rows = AK.mapreduce((x, y) -> x * y, +, a, b; init=0f0, dims=1)
```
"""
function mapreduce(
    f, op, src::MapReduceSource, backend::Backend=_mapreduce_backend(src);
    init,
    kwargs...
)
    _mapreduce_impl(
        f, op, src, backend;
        init,
        kwargs...
    )
end

function mapreduce(
    f, op, src::AbstractArray, src2::AbstractArray, srcs::AbstractArray...;
    init,
    kwargs...
)
    _mapreduce_check_map_axes(src, src2, srcs...)
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, src, src2, srcs...))
    return mapreduce(
        identity, op, bc, _mapreduce_backend(bc);
        init,
        kwargs...
    )
end


function _mapreduce_impl(
    f, op, src::MapReduceSource, backend::Backend;
    init,
    neutral=neutral_element(op, _mapreduce_eltype(src)),
    dims::Union{Nothing, Int, Tuple{Vararg{Int}}, Colon} = nothing,

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if isnothing(dims) || dims isa Colon
        if use_gpu_algorithm(backend, prefer_threads)
            mapreduce_1d_gpu(
                f, op, src, backend;
                init, neutral,
                max_tasks, min_elems,
                block_size, temp,
                switch_below
            )
        else
            mapreduce_1d_cpu(
                f, op, src, backend;
                init, neutral,
                max_tasks, min_elems,
                block_size, temp,
                switch_below
            )
        end
    else
        return mapreduce_nd(
            f, op, src, backend;
            init, neutral, dims,
            max_tasks, prefer_threads,
            min_elems, block_size,
            temp,
        )
    end
end
