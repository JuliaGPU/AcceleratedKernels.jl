# lo: rightmost 32 bits, hi: leftmost 32 bits
@inline _u32_lo(x::UInt64)::UInt32 = UInt32(x & UInt64(0xffffffff))
@inline _u32_hi(x::UInt64)::UInt32 = UInt32(x >> 32)

# leftmost 32 bits of a*b cast to UInt64s
@inline _mulhi_u32(a::UInt32, b::UInt32)::UInt32 = UInt32((UInt64(a) * UInt64(b)) >> 32)


@inline function _rotl32(x::UInt32, r::UInt32)::UInt32
    return (x << r) | (x >> (UInt32(32) - r))
end


@inline _counter_from_index(i)::UInt64 = UInt64(i - one(i))


@inline function rand_uint32(::AbstractCounterRNG, ::UInt64)::UInt32
    # Unrecognised AbstractCounterRNG
    throw(ArgumentError("No rand_uint32 implementation for this RNG type"))
end



"""
    uint32_to_unit_float32(u::UInt32) -> Float32

Convert a random `UInt32` to `Float32` in `[0, 1)` by mantissa construction.
"""
@inline function uint32_to_unit_float32(u::UInt32)::Float32
    # Keep 23 random bits for the mantissa (drop 9 rightmost bits from the UInt32)
    # and combine with the bit pattern of 1.0f0 (sign=0, exponent=127).
    bits = UInt32(0x3f800000) | (u >> 9)

    # Interpret as 1.mantissa, then subtract 1 for [0, 1)
    return reinterpret(Float32, bits) - 1.0f0
end
