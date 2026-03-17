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

Constructors:
- `CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())`
  Uses an explicit non-negative seed.
- `CounterRNG(; alg::CounterRNGAlgorithm=Philox())`
  Auto-seeds once using `Random.rand(Random.default_rng(), UInt64)`. Reusing the same `CounterRNG` instance is deterministic
  for fixed seed, algorithm, array shape, and eltype.
"""
struct CounterRNG{A <: CounterRNGAlgorithm} <: AbstractCounterRNG
    seed::UInt64
    alg::A
end


function CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())
    @argcheck seed >= 0 "Seed must be a non-negative integer"
    CounterRNG(UInt64(seed), alg)
end


function CounterRNG(; alg::CounterRNGAlgorithm=Philox())
    CounterRNG(Random.rand(Random.default_rng(), UInt64); alg)
end




# Shared helpers
include("utilities.jl")

# Algorithm-specific integer generators
include("splitmix64.jl")
include("philox.jl")
include("threefry.jl")




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
- `Bool`

Semantics:
- Unsigned integers: raw random bit patterns of requested width.
- Signed integers: corresponding unsigned patterns reinterpreted as signed.
- Floats: mantissa-based conversion from `UInt32`/`UInt64` into `[0, 1)`, uniform over the
  produced mantissa grid (not over all representable floats).
- Bool: `true` if the raw `UInt` draw is odd (`isodd(u)`), otherwise `false`.

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
    foreachindex(
        1:length(x), backend;
        max_tasks,
        min_elems,
        prefer_threads,
        block_size,
    ) do i
        @inbounds x[i] = rand_scalar(rng, _counter_from_index(i), T)
    end
    return x
end


function rand!(
    x::AbstractArray,
    args...;
    kwargs...,
)
    return rand!(CounterRNG(), x, args...; kwargs...)
end
