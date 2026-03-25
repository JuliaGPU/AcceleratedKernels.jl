abstract type CounterRNGAlgorithm end


"""
    CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox())

Counter-based RNG for [`rand!`](@ref).

`CounterRNG` stores:
- `seed`
- algorithm (`alg`)
- stream `offset`

The default algorithm is `Philox()`.

`seed` may be any non-negative `Integer`. It is normalised to `UInt64` internally.
`offset` is initialised to `0` by default and advances by `length(v)` after each [`rand!`](@ref)
call.

Constructors:
- `CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox(), offset::Integer=0)`
  Uses an explicit non-negative seed and offset.
- `CounterRNG(; alg::CounterRNGAlgorithm=Philox(), offset::Integer=0)`
  Auto-seeds once using `Base.rand(UInt64)`, with default `offset == 0`.
"""
mutable struct CounterRNG{A <: CounterRNGAlgorithm}
    const seed::UInt64
    const alg::A
    offset::UInt64
end


function CounterRNG(seed::Integer; alg::CounterRNGAlgorithm=Philox(), offset::Integer=0)
    @argcheck seed >= 0 "Seed must be a non-negative integer"
    @argcheck offset >= 0 "Offset must be a non-negative integer"
    CounterRNG(UInt64(seed), alg, UInt64(offset))
end


function CounterRNG(; alg::CounterRNGAlgorithm=Philox(), offset::Integer=0)
    CounterRNG(Base.rand(UInt64); alg, offset)
end


CounterRNG(seed::Integer, alg::CounterRNGAlgorithm) = CounterRNG(seed; alg)


"""
    reset!(rng::CounterRNG)

Reset `rng.offset` to `0x0`.
"""
@inline function reset!(rng::CounterRNG)
    rng.offset = UInt64(0)
    return rng
end




# Shared helpers
include("utilities.jl")

# Algorithm-specific integer generators
include("splitmix.jl")
include("philox.jl")
include("threefry.jl")

# Normally distributed scalar generators and randn!
include("randn.jl")




"""
    rand!(
        rng::CounterRNG,
        v::AbstractArray{T},
        backend::Backend=get_backend(v);

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # Implementation choice
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
    )

Fill `v` in-place with pseudo-random values using a counter-based RNG stream. For `v[i]`, the
counter is `rng.offset + UInt64(i - 1)` in linear indexing order.

After filling `v`, `rng.offset` advances by `length(v)`.

Supported scalar element types are:
- `UInt8`, `UInt16`, `UInt32`, `UInt64`
- `Int8`, `Int16`, `Int32`, `Int64`
- `Float16`, `Float32`, `Float64`
- `Bool`

Semantics:
- Unsigned integers: raw random bit patterns of requested width.
- Signed integers: corresponding unsigned patterns reinterpreted as signed.
- Floats: mantissa-based conversion from `UInt32`/`UInt64` into `[0, 1)`, uniform over the
  produced mantissa grid (not over all representable floats).
- Bool: `true` if the raw `UInt` draw is odd (`isodd(u)`), otherwise `false`.

"""
function rand!(
    rng::CounterRNG,
    v::AbstractArray{T},
    backend::Backend=get_backend(v);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
) where T

    @argcheck T <: ALLOWED_RAND_SCALARS "Unsupported eltype $T. Supported: $(ALLOWED_RAND_SCALARS)"

    # Local isbits captures from mutable rng object
    seed, alg, initial_offset = rng.seed, rng.alg, rng.offset
    
    foreachindex(
        v, backend;
        max_tasks, min_elems, prefer_threads, block_size,
    ) do i
        @inbounds v[i] = rand_scalar(seed, alg, initial_offset + _counter_from_index(i), T)
    end

    rng.offset += UInt64(length(v))
    
    v
end


function rand!(
    v::AbstractArray,
    args...;
    kwargs...,
)
    return rand!(CounterRNG(), v, args...; kwargs...)
end
