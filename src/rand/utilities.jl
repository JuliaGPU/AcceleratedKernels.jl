# lo: rightmost 32 bits, hi: leftmost 32 bits
@inline _u32_lo(x::UInt64)::UInt32 = UInt32(x & UInt64(0xffffffff))
@inline _u32_hi(x::UInt64)::UInt32 = UInt32(x >> 32)

# Construct UInt64 by bit concatenation of two UInt32s
@inline _u64_from_u32s(lo::UInt32, hi::UInt32)::UInt64 = (UInt64(hi) << 32) | UInt64(lo)

# Leftmost 32 bits of a*b cast to UInt64s
@inline _mulhi_u32(a::UInt32, b::UInt32)::UInt32 = UInt32((UInt64(a) * UInt64(b)) >> 32)

# 32-bit rotate left by r positions
@inline _rotl32(x::UInt32, r::UInt32)::UInt32 = bitrotate(x, Int32(r))

# Get counter used for CounterRNG from element index
@inline _counter_from_index(i)::UInt64 = UInt64(i - one(i))


# Shared allocation + fill helper for rand/randn convenience constructors.
@inline function _allocate_and_fill_rand(
    fill!,
    rng::CounterRNG,
    backend::Backend,
    ::Type{T},
    dims::Integer...;

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
) where {T}
    dims_int = Base.map(Int, dims)
    v = KernelAbstractions.allocate(backend, T, dims_int)
    fill!(rng, v, backend; max_tasks, min_elems, prefer_threads, block_size)
    return v
end




@inline _rand_scalar_uint_type(::Type{UInt8}) = UInt32
@inline _rand_scalar_uint_type(::Type{UInt16}) = UInt32
@inline _rand_scalar_uint_type(::Type{UInt32}) = UInt32
@inline _rand_scalar_uint_type(::Type{Int8}) = UInt32
@inline _rand_scalar_uint_type(::Type{Int16}) = UInt32
@inline _rand_scalar_uint_type(::Type{Int32}) = UInt32
@inline _rand_scalar_uint_type(::Type{Float16}) = UInt32
@inline _rand_scalar_uint_type(::Type{Float32}) = UInt32
@inline _rand_scalar_uint_type(::Type{UInt64}) = UInt64
@inline _rand_scalar_uint_type(::Type{Int64}) = UInt64
@inline _rand_scalar_uint_type(::Type{Float64}) = UInt64
@inline _rand_scalar_uint_type(::Type{Bool}) = UInt32


@inline _rand_scalar_from_uint(::Type{UInt8}, u::UInt32)::UInt8 = trunc(UInt8, u >> 24)
@inline _rand_scalar_from_uint(::Type{UInt16}, u::UInt32)::UInt16 = trunc(UInt16, u >> 16)
@inline _rand_scalar_from_uint(::Type{UInt32}, u::UInt32)::UInt32 = u
@inline _rand_scalar_from_uint(::Type{UInt64}, u::UInt64)::UInt64 = u
@inline _rand_scalar_from_uint(::Type{Int8}, u::UInt32)::Int8 = reinterpret(Int8, trunc(UInt8, u >> 24))
@inline _rand_scalar_from_uint(::Type{Int16}, u::UInt32)::Int16 = reinterpret(Int16, trunc(UInt16, u >> 16))
@inline _rand_scalar_from_uint(::Type{Int32}, u::UInt32)::Int32 = reinterpret(Int32, u)
@inline _rand_scalar_from_uint(::Type{Int64}, u::UInt64)::Int64 = reinterpret(Int64, u)
@inline _rand_scalar_from_uint(::Type{Float16}, u::UInt32)::Float16 = uint32_to_unit_float16(u)
@inline _rand_scalar_from_uint(::Type{Float32}, u::UInt32)::Float32 = uint32_to_unit_float32(u)
@inline _rand_scalar_from_uint(::Type{Float64}, u::UInt64)::Float64 = uint64_to_unit_float64(u)
@inline _rand_scalar_from_uint(::Type{Bool}, u::UInt32)::Bool = isodd(u)


#=
Every RNG algorithm implements rand_uint(seed, alg, counter, UInt32/UInt64).
This is the fallback for unsupported RNG algorithms.
=#
"""
    rand_uint(seed::UInt64, alg::CounterRNGAlgorithm, counter::UInt64, ::Type{UIntType}) -> UIntType
    where {UIntType <: Union{UInt32, UInt64}}

Low-level extension point for counter-based RNG algorithms used by [`CounterRNG`](@ref).

`rand_uint` must deterministically map `(seed, alg, counter)` to a raw unsigned integer of the
requested width. Custom algorithms should implement methods for both:

- `rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt32})::UInt32`
- `rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt64})::UInt64`

These methods are used internally by [`rand!`](@ref), [`rand`](@ref), [`randn!`](@ref), and
[`randn`](@ref) to generate integers, floats, and normal samples.

# Requirements
- The mapping must be deterministic for fixed `seed`, `alg`, and `counter`.
- Implement both `UInt32` and `UInt64` widths.
- The method should return raw random bits; higher-level type conversion is handled by AK separately.

# Notes
- `counter` is the logical stream position (typically the array index).
- For block-based algorithms such as Philox or Threefry, the `UInt32` and `UInt64` methods may
  share an internal block computation.
- The fallback method throws an `ArgumentError` for algorithms that do not implement `rand_uint`.

See also: [`CounterRNGAlgorithm`](@ref), [`CounterRNG`](@ref).
"""
@inline function rand_uint(
    ::UInt64,
    alg::CounterRNGAlgorithm,
    ::UInt64,
    ::Type{UIntType}
)::UIntType where {UIntType <: Union{UInt32, UInt64}}
    throw(ArgumentError("No rand_uint implementation for RNG algorithm: $(typeof(alg))"))
end


#=
Shared scalar generation:
1) map requested scalar type to corresponding raw UInt width
2) fill the UInt with random bits
3) convert bits into requested scalar representation
=#
@inline function rand_scalar(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{T}
)::T where {T <: ALLOWED_RAND_SCALARS}

    UIntType = _rand_scalar_uint_type(T)
    u = rand_uint(seed, alg, counter, UIntType)

    return _rand_scalar_from_uint(T, u)
end


@inline function rand_scalar(::UInt64, ::CounterRNGAlgorithm, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported random scalar type $(T). Supported: $(ALLOWED_RAND_SCALARS)"
    ))
end


# Convert random UInt32 bits to Float16 in [0, 1) by mantissa construction.
@inline function uint32_to_unit_float16(u::UInt32)::Float16

    # Keep 10 random bits for the mantissa (drop 22 rightmost bits from the UInt32)
    # and combine with the bit pattern of Float16(1.0) (sign=0, exponent=15).
    bits = UInt16(0x3c00) | UInt16(u >> 22)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    reinterpret(Float16, bits) - Float16(1)
end


# Convert random UInt32 bits to Float32 in [0, 1) by mantissa construction.
@inline function uint32_to_unit_float32(u::UInt32)::Float32

    # Keep 23 random bits for the mantissa (drop 9 rightmost bits from the UInt32)
    # and combine with the bit pattern of 1.0f0 (sign=0, exponent=127).
    bits = UInt32(0x3f800000) | (u >> 9)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    reinterpret(Float32, bits) - 1.0f0
end


# Convert random UInt64 bits to Float64 in [0, 1) by mantissa construction.
@inline function uint64_to_unit_float64(u::UInt64)::Float64

    # Keep 52 random bits for the mantissa (drop 12 rightmost bits from the UInt64)
    # and combine with the bit pattern of 1.0 (sign=0, exponent=1023).
    bits = UInt64(0x3ff0000000000000) | (u >> 12)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    reinterpret(Float64, bits) - 1.0
end





### Helpers for randn ###


# Midpoint-mapped open-interval Float sampling in (0, 1), used for Box-Muller
const OPEN01_MAX_MIDPOINT_INDEX_F32 = UInt32(0x00fffffe)
const OPEN01_MAX_MIDPOINT_INDEX_F64 = UInt64(0x001ffffffffffffe)
const OPEN01_MIDPOINT_SCALE_F32 = ldexp(Float32(1), -24)
const OPEN01_MIDPOINT_SCALE_F64 = ldexp(Float64(1), -53)


# Convert random UInt32 bits to Float32 in (0, 1) using midpoint mapping on a 24-bit grid.
@inline function _uint32_to_open_unit_float32_midpoint(u::UInt32)::Float32
    # `min` keeps the top midpoint below one after Float32 rounding.
    k = min(u >> 8, OPEN01_MAX_MIDPOINT_INDEX_F32)
    return (Float32(k) + 0.5f0) * OPEN01_MIDPOINT_SCALE_F32
end


# Convert random UInt64 bits to Float64 in (0, 1) using midpoint mapping on a 53-bit grid.
@inline function _uint64_to_open_unit_float64_midpoint(u::UInt64)::Float64
    # `min` keeps the top midpoint below one after Float64 rounding.
    k = min(u >> 11, OPEN01_MAX_MIDPOINT_INDEX_F64)
    return (Float64(k) + 0.5) * OPEN01_MIDPOINT_SCALE_F64
end


# Float16 path reuses Float32 midpoint sampling for robust math in Box-Muller.
@inline function rand_float_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float16},
)::Float16
    return Float16(rand_float_open01(seed, alg, counter, Float32))
end


@inline function rand_float_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float32},
)::Float32
    return _uint32_to_open_unit_float32_midpoint(rand_uint(seed, alg, counter, UInt32))
end


@inline function rand_float_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float64},
)::Float64
    return _uint64_to_open_unit_float64_midpoint(rand_uint(seed, alg, counter, UInt64))
end


@inline function rand_float_open01(::UInt64, ::CounterRNGAlgorithm, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported open-interval random type $(T). Supported: Union{Float16, Float32, Float64}"
    ))
end
