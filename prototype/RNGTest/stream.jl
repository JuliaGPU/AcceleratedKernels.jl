import AcceleratedKernels as AK


function make_rng(seed::Integer, alg::Symbol; offset::Integer=0)
    if alg === :philox
        return AK.CounterRNG(seed; alg=AK.Philox(), offset=offset)
    elseif alg === :threefry
        return AK.CounterRNG(seed; alg=AK.Threefry(), offset=offset)
    elseif alg === :splitmix64
        return AK.CounterRNG(seed; alg=AK.SplitMix64(), offset=offset)
    end
    throw(ArgumentError("alg must be :philox, :threefry, or :splitmix64; got $alg"))
end


mutable struct AKUInt64Stream{R}
    rng::R
    chunk::Int
    idx::Int
    host_scratch::Vector{UInt64}
    refill_count::Int
end


function AKUInt64Stream(
    host_scratch::Vector{UInt64};
    seed::Integer=0x1234,
    alg::Symbol=:philox,
    start_counter::UInt64=0x0000000000000000,
)
    chunk = length(host_scratch)
    chunk > 0 || throw(ArgumentError("host_scratch must be non-empty"))
    rng = make_rng(seed, alg; offset=start_counter)

    return AKUInt64Stream(
        rng,
        chunk,
        chunk + 1,
        host_scratch,
        0,
    )
end


@inline _u01_from_u64(u::UInt64)::Float64 = Float64(u >>> 11) * 0x1.0p-53


function _fill_chunk!(s::AKUInt64Stream)
    AK.rand!(s.rng, s.host_scratch)
    return nothing
end


function refill!(s::AKUInt64Stream)
    _fill_chunk!(s)
    s.idx = 1
    s.refill_count += 1
    return s
end


function next_u64!(s::AKUInt64Stream)::UInt64
    if s.idx > s.chunk
        refill!(s)
    end
    @inbounds u = s.host_scratch[s.idx]
    s.idx += 1
    return u
end


@inline next_float64!(s::AKUInt64Stream)::Float64 = _u01_from_u64(next_u64!(s))


function make_rngtest_generator!(s::AKUInt64Stream)
    if s.idx > s.chunk
        refill!(s)
    end
    return () -> next_float64!(s)
end
