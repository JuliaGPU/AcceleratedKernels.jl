# neutral_element moved over from GPUArrays.jl
neutral_element(op, T) =
    error("""AcceleratedKernels.jl needs to know the neutral element for your operator `$op`.
             Please pass it as an explicit keyword argument `neutral`.""")
neutral_element(::typeof(Base.:(|)), T) = zero(T)
neutral_element(::typeof(Base.:(+)), T) = zero(T)
neutral_element(::typeof(Base.add_sum), T) = zero(T)
neutral_element(::typeof(Base.:(&)), T) = one(T)
neutral_element(::typeof(Base.:(*)), T) = one(T)
neutral_element(::typeof(Base.mul_prod), T) = one(T)
neutral_element(::typeof(Base.min), T) = typemax(T)
neutral_element(::typeof(Base.max), T) = typemin(T)
neutral_element(::typeof(Base._extrema_rf), ::Type{<:NTuple{2,T}}) where {T} = typemax(T), typemin(T)


include("mapreduce_1d.jl")
include("mapreduce_nd.jl")


"""
    reduce(
        op, src::AbstractArray, backend::Backend=get_backend(src);
        init,
        neutral=neutral_element(op, eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        scheduler=:static,
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Reduce `src` along dimensions `dims` using the binary operator `op`. If `dims` is `nothing`, reduce
`src` to a scalar. If `dims` is an integer, reduce `src` along that dimension. The `init` value is
used as the initial value for the reduction; `neutral` is the neutral element for the operator `op`.

## CPU settings
The `scheduler` can be one of the [OhMyThreads.jl schedulers](https://juliafolds2.github.io/OhMyThreads.jl/dev/refs/api/#Schedulers),
i.e. `:static`, `:dynamic`, `:greedy` or `:serial`. Assuming the workload is uniform (as the GPU
algorithm prefers), `:static` is used by default; if you need fine-grained control over your
threads, consider using [`OhMyThreads.jl`](https://github.com/JuliaFolds2/OhMyThreads.jl) directly.

Use at most `max_tasks` threads with at least `min_elems` elements per task.

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

# Platform-Specific Notes
N-dimensional reductions on the CPU are not parallel yet ([issue](https://github.com/JuliaFolds2/OhMyThreads.jl/issues/128)),
and defer to `Base.reduce`.

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
    _reduce_impl(
        op, src, backend;
        init,
        kwargs...
    )
end


function _reduce_impl(
    op, src::AbstractArray, backend;
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
        scheduler=:static,
        max_tasks=Threads.nthreads(),
        min_elems=1,

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

## CPU settings
The `scheduler` can be one of the [OhMyThreads.jl schedulers](https://juliafolds2.github.io/OhMyThreads.jl/dev/refs/api/#Schedulers),
i.e. `:static`, `:dynamic`, `:greedy` or `:serial`. Assuming the workload is uniform (as the GPU
algorithm prefers), `:static` is used by default; if you need fine-grained control over your
threads, consider using [`OhMyThreads.jl`](https://github.com/JuliaFolds2/OhMyThreads.jl) directly.

Use at most `max_tasks` threads with at least `min_elems` elements per task.

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
    scheduler=:static,
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if backend isa GPU
        if isnothing(dims)
            return mapreduce_1d(
                f, op, src, backend;
                init, neutral,
                block_size, temp,
                switch_below,
            )
        else
            return mapreduce_nd(
                f, op, src, backend;
                init, neutral,
                dims, block_size,
                temp,
            )
        end
    else
        if isnothing(dims)
            num_elems = length(src)
            num_tasks = min(max_tasks, num_elems ÷ min_elems)
            if num_tasks <= 1
                return Base.mapreduce(f, op, src; init)
            end
            return op(init, OMT.tmapreduce(
                f, op, src; init=neutral,
                scheduler=scheduler,
                outputtype=typeof(init),
                nchunks=num_tasks,
            ))
        else
            # FIXME: waiting on OhMyThreads.jl for n-dimensional reduction
            return Base.mapreduce(f, op, src; init, dims)
        end
    end
end
