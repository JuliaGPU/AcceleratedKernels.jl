# Backend implementations
include("utilities.jl")
include("mapreduce_1d_cpu.jl")
include("mapreduce_1d_gpu.jl")
include("mapreduce_nd.jl")


"""
    reduce(
        op, src::AbstractArray, backend::Backend=get_backend(src);
        init,
        neutral=neutral_element(op, eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op`. If `dims` is `nothing`, reduce
`src` to a scalar. If `dims` is an integer, reduce `src` along that dimension. The `init` value is
used as the initial value for the reduction; `neutral` is the neutral element for the operator `op`.

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
(`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
zero, check against `Base.reduce` for CPU arrays for exact behavior.

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
    op, src::AbstractArray, backend::Backend=get_backend(src);
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
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op` after applying `f` elementwise.
If `dims` is `nothing`, reduce `src` to a scalar. If `dims` is an integer, reduce `src` along that
dimension. The `init` value is used as the initial value for the reduction (i.e. after mapping).

The `neutral` value is the neutral element (zero) for the operator `op`, which is needed for an
efficient GPU implementation that also allows a nonzero `init`.

The returned type is the same as `init` - to control output precision, specify `init` explicitly.

## CPU settings
Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
cases are scaling linearly with the number of threads.

## GPU settings
The `block_size` parameter controls the number of threads per block.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
zero, check against `Base.reduce` for CPU arrays for exact behavior.

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
"""
function mapreduce(
    f, op, src::AbstractArray, backend::Backend=get_backend(src);
    init,
    kwargs...
)
    _mapreduce_impl(
        f, op, src, backend;
        init,
        kwargs...
    )
end


function _mapreduce_impl(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(src)),
    dims::Union{Nothing, Int}=nothing,

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if isnothing(dims)
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
