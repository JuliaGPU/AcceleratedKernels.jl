"""
    abstract type AbstractCounterRNG end
    abstract type CounterRNGAlgorithm end

RNG interface for counter-based random generation with AcceleratedKernels.
"""

abstract type AbstractCounterRNG end
abstract type CounterRNGAlgorithm end


"""
    CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())

Stateless counter-based RNG configuration for [`rand!`](@ref).

`CounterRNG` is immutable and does not hold mutable thread-local or global state. Each generated
value is a pure function of:
- `seed`
- logical linear element index
- algorithm (`alg`)

The default algorithm is `Philox()`.

`seed` may be any non-negative `Integer`. It is normalised to `UInt64` internally.
"""
struct CounterRNG{A <: CounterRNGAlgorithm} <: AbstractCounterRNG
    seed::UInt64
    alg::A
end


function CounterRNG(seed::Unsigned; alg::CounterRNGAlgorithm=Philox())
    CounterRNG(UInt64(seed), alg)
end


function CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())
    @argcheck seed >= 0
    CounterRNG(UInt64(seed), alg)
end



"""
    CounterRNG(; alg::CounterRNGAlgorithm=SplitMix64())

Create a stateless counter-based RNG with an automatically generated seed.

The seed is sampled exactly once at construction using `rand(UInt64)`. Reusing this same
`CounterRNG` instance is deterministic for fixed seed, algorithm, array shape, and eltype.
"""
function CounterRNG(; alg::CounterRNGAlgorithm=SplitMix64())
    CounterRNG(Base.rand(UInt64); alg)
end




# Shared helpers
include("utilities.jl")

# Algorithm-specific integer generators
include("splitmix64.jl")
include("philox.jl")
include("threefry.jl")





function _rand_fill_threads!(
    rng::AbstractCounterRNG,
    x::AbstractArray{T};
    max_tasks::Int,
    min_elems::Int,
) where {T <: ALLOWED_RAND_SCALARS}
    task_partition(length(x), max_tasks, min_elems) do irange
        @inbounds for i in irange
            counter = _counter_from_index(i)
            x[i] = rand_scalar(rng, counter, T)
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
        x[i] = rand_scalar(rng, counter, eltype(x))
    end
end


function _rand_fill_gpu!(
    rng::AbstractCounterRNG,
    x::AbstractArray{T},
    backend::Backend;
    block_size::Int,
) where {T <: ALLOWED_RAND_SCALARS}
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
        x::AbstractArray{T},
        backend::Backend=get_backend(x);

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # Implementation choice
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
    )

Fill `x` in-place with pseudo-random values using a stateless counter-based RNG. For `x[i]`, the
counter is exactly `UInt64(i - 1)` in linear indexing order.

Supported scalar element types are:
- `UInt32`, `UInt64`
- `Int32`, `Int64`
- `Float32`, `Float64`

Semantics:
- Unsigned integers: raw random bit patterns of requested width.
- Signed integers: corresponding unsigned patterns reinterpreted as signed.
- Floats: mantissa-based conversion from `UInt32`/`UInt64` into `[0, 1)`, uniform over the
  produced mantissa grid (not over all representable floats).
"""
function rand!(
    rng::AbstractCounterRNG,
    x::AbstractArray{T},
    backend::Backend=get_backend(x);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
) where T

    @argcheck T <: ALLOWED_RAND_SCALARS "Unsupported eltype $T. Supported: $(ALLOWED_RAND_SCALARS)"

    if use_gpu_algorithm(backend, prefer_threads)
        return _rand_fill_gpu!(rng, x, backend; block_size)
    else
        return _rand_fill_threads!(rng, x; max_tasks, min_elems)
    end
end


"""
    rand!(
        x::AbstractArray{T},
        args...;
        kwargs...,
    )

Convenience overload that creates a fresh `CounterRNG()` and fills `x`.

Each call to `rand!(x, ...)` auto-seeds a new RNG once using `rand(UInt64)`, so repeated calls
produce different outputs unless an explicit `CounterRNG` is provided.
"""
function rand!(
    x::AbstractArray,
    args...;
    kwargs...,
)
    return rand!(CounterRNG(), x, args...; kwargs...)
end
