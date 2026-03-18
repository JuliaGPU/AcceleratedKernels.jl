struct SplitMix64 <: CounterRNGAlgorithm end

# SplitMix64 magic numbers
const SPLITMIX64_INCREMENT = UInt64(0x9e3779b97f4a7c15)
const SPLITMIX64_MIX_A = UInt64(0xbf58476d1ce4e5b9)
const SPLITMIX64_MIX_B = UInt64(0x94d049bb133111eb)


@inline function _splitmix64_mix(x::UInt64)::UInt64
    x = xor(x, x >> 30)
    x *= SPLITMIX64_MIX_A
    x = xor(x, x >> 27)
    x *= SPLITMIX64_MIX_B
    x = xor(x, x >> 31)
    return x
end


# Derive a 32-bit seed word from a 64-bit seed using SplitMix64 mixing.
@inline function splitmix32_from_u64(seed::UInt64)::UInt32
    return _u32_hi(_splitmix64_mix(seed + SPLITMIX64_INCREMENT))
end


# Natural SplitMix64 output path: compute 64 random bits directly from one counter
@inline function rand_uint(
    rng::CounterRNG{<:SplitMix64},
    counter::UInt64,
    ::Type{UInt64},
)::UInt64
    seed = UInt64(rng.seed)
    return _splitmix64_mix(counter + seed + SPLITMIX64_INCREMENT)
end


# UInt32 path is derived from the high 32 bits of the UInt64 SplitMix output
@inline function rand_uint(
    rng::CounterRNG{<:SplitMix64},
    counter::UInt64,
    ::Type{UInt32},
)::UInt32
    return _u32_hi(rand_uint(rng, counter, UInt64))
end
