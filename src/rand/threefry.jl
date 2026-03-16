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


# Evaluate one Threefry block at `counter`, returning two 32-bit lanes `(x0, x1)`
@inline function _threefry2x32_block(
    rng::CounterRNG{<:Threefry},
    counter::UInt64,
)::Tuple{UInt32, UInt32}
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

    return x0, x1
end


# Return lane 0 from the single Threefry block at `counter`
@inline function rand_uint(
    rng::CounterRNG{<:Threefry},
    counter::UInt64,
    ::Type{UInt32},
)::UInt32
    x0, _ = _threefry2x32_block(rng, counter)
    return x0
end


# Build UInt64 from the two lanes `(x0, x1)` of the same Threefry block at `counter`
@inline function rand_uint(
    rng::CounterRNG{<:Threefry},
    counter::UInt64,
    ::Type{UInt64},
)::UInt64
    x0, x1 = _threefry2x32_block(rng, counter)
    return _u64_from_u32s(x0, x1)
end
