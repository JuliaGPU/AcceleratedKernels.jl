"""
    abstract type AbstractCounterRNG end
    abstract type CounterRNGAlgorithm end

RNG interface for counter-based random generation with AcceleratedKernels.
"""

abstract type AbstractCounterRNG end
abstract type CounterRNGAlgorithm end


"""
    CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=SplitMix64())

Stateless counter-based RNG configuration for [`rand!`](@ref).

`CounterRNG` is immutable and does not hold mutable thread-local or global state. Each generated
value is a pure function of:
- `seed`
- logical linear element index
- algorithm (`alg`)

The default algorithm is `Philox()`.
"""
struct CounterRNG{K <: Unsigned, A <: CounterRNGAlgorithm} <: AbstractCounterRNG
    seed::K
    alg::A
end


function CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())
    @argcheck seed >= 0
    CounterRNG(UInt64(seed), alg)
end




# Shared helpers
include("utilities.jl")

# Algorithm-specific integer generators
include("splitmix64.jl")
include("philox.jl")
include("threefry.jl")





function _rand_fill_threads!(
    rng::AbstractCounterRNG,
    x::AbstractArray{Float32};
    max_tasks::Int,
    min_elems::Int,
)
    task_partition(length(x), max_tasks, min_elems) do irange
        @inbounds for i in irange
            counter = _counter_from_index(i)
            x[i] = uint32_to_unit_float32(rand_uint32(rng, counter))
        end
    end
    return x
end


@kernel inbounds=true cpu=false unsafe_indices=true function _rand_fill_kernel!(
    rng,
    x,
)
    i = @index(Global, Linear)
    if i <= length(x)
        counter = _counter_from_index(i)
        x[i] = uint32_to_unit_float32(rand_uint32(rng, counter))
    end
end


function _rand_fill_gpu!(
    rng::AbstractCounterRNG,
    x::AbstractArray{Float32},
    backend::Backend;
    block_size::Int,
)
    @argcheck block_size > 0
    len = length(x)
    len == 0 && return x

    blocks = div(len, block_size, RoundUp)
    kernel! = _rand_fill_kernel!(backend, block_size)
    kernel!(rng, x, ndrange=(blocks * block_size,))
    return x
end


"""
    rand!(
        rng::AbstractCounterRNG,
        x::AbstractArray{Float32},
        backend::Backend=get_backend(x);

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # Implementation choice
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
    )

Fill `x` in-place with pseudo-random `Float32` values in `[0, 1)` using a stateless counter-based
RNG. For `x[i]`, the counter is exactly `UInt64(i - 1)` in linear indexing order.

The float conversion is mantissa-based: uniform over the produced mantissa grid, not over all
representable `Float32` values in `[0, 1)`.
"""
function rand!(
    rng::AbstractCounterRNG,
    x::AbstractArray{Float32},
    backend::Backend=get_backend(x);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
)
    if use_gpu_algorithm(backend, prefer_threads)
        _rand_fill_gpu!(rng, x, backend; block_size)
    else
        _rand_fill_threads!(rng, x; max_tasks, min_elems)
    end
end
