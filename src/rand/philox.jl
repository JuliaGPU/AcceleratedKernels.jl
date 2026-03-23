struct Philox <: CounterRNGAlgorithm end


# Philox magic numbers
const PHILOX_M0 = UInt32(0xD256D193)
const PHILOX_W0 = UInt32(0x9E3779B9)
const PHILOX_ROUNDS = 10


@inline function _philox2x32_round(x0::UInt32, x1::UInt32, k0::UInt32)
    lo = PHILOX_M0 * x0
    hi = _mulhi_u32(PHILOX_M0, x0)
    y0 = xor(xor(hi, k0), x1)
    y1 = lo
    return y0, y1
end


# Evaluate one Philox block at `counter`, returning two 32-bit lanes `(x0, x1)`
@inline function _philox2x32_block(
    seed::UInt64,
    counter::UInt64,
)::Tuple{UInt32, UInt32}
    x0 = _u32_lo(counter)
    x1 = _u32_hi(counter)

    k0 = splitmix32_from_u64(seed)

    @inbounds for _ in 1:PHILOX_ROUNDS
        x0, x1 = _philox2x32_round(x0, x1, k0)
        k0 += PHILOX_W0
    end

    return x0, x1
end


# Return lane 0 from the single Philox block at `counter`
@inline function rand_uint(
    seed::UInt64,
    alg::Philox,
    counter::UInt64,
    ::Type{UInt32},
)::UInt32
    x0, _ = _philox2x32_block(seed, counter)
    return x0
end


# Build UInt64 from the two lanes `(x0, x1)` of the same Philox block at `counter`
@inline function rand_uint(
    seed::UInt64,
    alg::Philox,
    counter::UInt64,
    ::Type{UInt64},
)::UInt64
    x0, x1 = _philox2x32_block(seed, counter)
    return _u64_from_u32s(x0, x1)
end
