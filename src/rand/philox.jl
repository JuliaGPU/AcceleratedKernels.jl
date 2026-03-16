struct Philox <: CounterRNGAlgorithm end


# Philox magic numbers
const PHILOX_M0 = UInt32(0xD256D193)
const PHILOX_W0 = UInt32(0x9E3779B9)
const PHILOX_ROUNDS = 10


# Each round destroys x0 with multiplication, addition, and XORs
@inline function _philox2x32_round(x0::UInt32, x1::UInt32, k0::UInt32)
    lo = PHILOX_M0 * x0
    hi = _mulhi_u32(PHILOX_M0, x0)
    y0 = xor(xor(hi, k0), x1)
    y1 = lo
    return y0, y1
end


"""
    rand_uint32(rng::CounterRNG{<:Unsigned, Philox}, counter::UInt64) -> UInt32
"""
@inline function rand_uint32(rng::CounterRNG{<:Unsigned, Philox}, counter::UInt64)::UInt32
    x0 = _u32_lo(counter)
    x1 = _u32_hi(counter)

    seed = UInt64(rng.seed)
    k0 = _u32_lo(seed)
    x1 = xor(x1, _u32_hi(seed))

    @inbounds for _ in 1:PHILOX_ROUNDS
        x0, x1 = _philox2x32_round(x0, x1, k0)
        k0 += PHILOX_W0
    end

    return x0
end
