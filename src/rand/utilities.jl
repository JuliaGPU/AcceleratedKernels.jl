# lo: rightmost 32 bits, hi: leftmost 32 bits
@inline _u32_lo(x::UInt64)::UInt32 = UInt32(x & UInt64(0xffffffff))
@inline _u32_hi(x::UInt64)::UInt32 = UInt32(x >> 32)
@inline _u64_from_u32s(lo::UInt32, hi::UInt32)::UInt64 = (UInt64(hi) << 32) | UInt64(lo)

# leftmost 32 bits of a*b cast to UInt64s
@inline _mulhi_u32(a::UInt32, b::UInt32)::UInt32 = UInt32((UInt64(a) * UInt64(b)) >> 32)

# 32-bit rotate left by r positions
@inline _rotl32(x::UInt32, r::UInt32)::UInt32 = (x << r) | (x >> (UInt32(32) - r))


@inline _counter_from_index(i)::UInt64 = UInt64(i - one(i))


# Internal scalar eltypes currently supported by rand!.
const ALLOWED_RAND_SCALARS = Union{
    UInt32, UInt64,
    Int32, Int64,
    Float32, Float64,
}


@inline raw_uint_type(::Type{UInt32}) = UInt32
@inline raw_uint_type(::Type{Int32}) = UInt32
@inline raw_uint_type(::Type{Float32}) = UInt32
@inline raw_uint_type(::Type{UInt64}) = UInt64
@inline raw_uint_type(::Type{Int64}) = UInt64
@inline raw_uint_type(::Type{Float64}) = UInt64


@inline from_uint(::Type{UInt32}, u::UInt32)::UInt32 = u
@inline from_uint(::Type{UInt64}, u::UInt64)::UInt64 = u
@inline from_uint(::Type{Int32}, u::UInt32)::Int32 = reinterpret(Int32, u)
@inline from_uint(::Type{Int64}, u::UInt64)::Int64 = reinterpret(Int64, u)
@inline from_uint(::Type{Float32}, u::UInt32)::Float32 = uint32_to_unit_float32(u)
@inline from_uint(::Type{Float64}, u::UInt64)::Float64 = uint64_to_unit_float64(u)


#=
Every RNG algorithm implements rand_uint(rng, counter, UInt32/UInt64).
This fallback provides a clear failure for unsupported RNG types.
=#
@inline function rand_uint(
    rng::AbstractCounterRNG,
    ::UInt64,
    ::Type{UIntType}
)::UIntType where {UIntType <: Union{UInt32, UInt64}}
    throw(ArgumentError("No rand_uint implementation for RNG: $rng"))
end


#=
Shared scalar generation:
1) map requested scalar type to corresponding raw UInt width
2) fill the UInt with random bits
3) convert bits into requested scalar representation
=#
@inline function rand_scalar(
    rng::AbstractCounterRNG,
    counter::UInt64,
    ::Type{T}
)::T where {T <: ALLOWED_RAND_SCALARS}

    UIntType = raw_uint_type(T)
    u = rand_uint(rng, counter, UIntType)

    return from_uint(T, u)
end


@inline function rand_scalar(::AbstractCounterRNG, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported random scalar type $(T). Supported: UInt32, UInt64, Int32, Int64, Float32, Float64."
    ))
end




# Convert random UInt32 bits to Float32 in [0, 1) by mantissa construction.
@inline function uint32_to_unit_float32(u::UInt32)::Float32
    # Keep 23 random bits for the mantissa (drop 9 rightmost bits from the UInt32)
    # and combine with the bit pattern of 1.0f0 (sign=0, exponent=127).
    bits = UInt32(0x3f800000) | (u >> 9)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    return reinterpret(Float32, bits) - 1.0f0
end


# Convert random UInt64 bits to Float64 in [0, 1) by mantissa construction.
@inline function uint64_to_unit_float64(u::UInt64)::Float64
    # Keep 52 random bits for the mantissa (drop 12 rightmost bits from the UInt64)
    # and combine with the bit pattern of 1.0 (sign=0, exponent=1023).
    bits = UInt64(0x3ff0000000000000) | (u >> 12)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    return reinterpret(Float64, bits) - 1.0
end
