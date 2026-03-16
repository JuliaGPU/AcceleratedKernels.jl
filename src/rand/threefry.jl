struct Threefry <: CounterRNGAlgorithm end

# Threefry magic numbers
const THREEFRY_PARITY = UInt32(0x1BD11BDA)
const THREEFRY_ROTATIONS = (
    UInt32(13), UInt32(15), UInt32(26), UInt32(6),
    UInt32(17), UInt32(29), UInt32(16), UInt32(24),
)
const THREEFRY_ROUNDS = 20


@inline function _threefry_key_word(k0::UInt32, k1::UInt32, k2::UInt32, idx::Int)::UInt32
    idx == 0 && return k0
    idx == 1 && return k1
    return k2
end


"""
    rand_uint32(rng::CounterRNG{<:Unsigned, Threefry}, counter::UInt64) -> UInt32
"""
@inline function rand_uint32(rng::CounterRNG{<:Unsigned, Threefry}, counter::UInt64)::UInt32
    x0 = _u32_lo(counter)
    x1 = _u32_hi(counter)

    seed = UInt64(rng.seed)
    k0 = _u32_lo(seed)
    k1 = _u32_hi(seed)
    k2 = xor(THREEFRY_PARITY, xor(k0, k1))

    x0 += k0
    x1 += k1

    @inbounds for round in 0:(THREEFRY_ROUNDS - 1)
        rot = THREEFRY_ROTATIONS[(round & 0x7) + 1]
        x0 += x1
        x1 = xor(_rotl32(x1, rot), x0)

        if (round & 0x3) == 3
            s = (round >>> 2) + 1
            i0 = s % 3
            i1 = (s + 1) % 3
            x0 += _threefry_key_word(k0, k1, k2, i0)
            x1 += _threefry_key_word(k0, k1, k2, i1) + UInt32(s)
        end
    end

    return x0
end
