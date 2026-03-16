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


"""
    rand_uint(rng::CounterRNG{<:SplitMix64}, counter::UInt64, UInt64) -> UInt64
"""
@inline function rand_uint(
    rng::CounterRNG{<:SplitMix64},
    counter::UInt64,
    ::Type{UInt64},
)::UInt64
    seed = UInt64(rng.seed)
    return _splitmix64_mix(counter + seed + SPLITMIX64_INCREMENT)
end


"""
    rand_uint(rng::CounterRNG{<:SplitMix64}, counter::UInt64, UInt32) -> UInt32
"""
@inline function rand_uint(
    rng::CounterRNG{<:SplitMix64},
    counter::UInt64,
    ::Type{UInt32},
)::UInt32
    return _u32_hi(rand_uint(rng, counter, UInt64))
end
