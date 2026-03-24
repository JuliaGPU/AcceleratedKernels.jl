# lo: rightmost 32 bits, hi: leftmost 32 bits
@inline _u32_lo(x::UInt64)::UInt32 = UInt32(x & UInt64(0xffffffff))
@inline _u32_hi(x::UInt64)::UInt32 = UInt32(x >> 32)

# Construct UInt64 by bit concatenation of two UInt32s
@inline _u64_from_u32s(lo::UInt32, hi::UInt32)::UInt64 = (UInt64(hi) << 32) | UInt64(lo)

# Leftmost 32 bits of a*b cast to UInt64s
@inline _mulhi_u32(a::UInt32, b::UInt32)::UInt32 = UInt32((UInt64(a) * UInt64(b)) >> 32)

# 32-bit rotate left by r positions
@inline _rotl32(x::UInt32, r::UInt32)::UInt32 = (x << r) | (x >> (UInt32(32) - r))

# Get counter used for CounterRNG from element index
@inline _counter_from_index(i)::UInt64 = UInt64(i - one(i))


# Internal scalar eltypes currently supported by rand!.
const ALLOWED_RAND_SCALARS = Union{
    UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float16, Float32, Float64,
    Bool
}


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
