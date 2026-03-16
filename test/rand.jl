function _is_unit_interval(v)
    for x in v
        if isnan(x) || x < 0.0f0 || x >= 1.0f0
            return false
        end
    end
    return true
end

function _rand_fill_reference!(rng, x::AbstractArray{Float32})
    @inbounds for i in eachindex(x)
        counter = UInt64(i - one(i))
        x[i] = AK.uint32_to_unit_float32(AK.rand_uint32(rng, counter))
    end
    return x
end

@testset "rand" begin
    @test AK.CounterRNG{AK.SplitMix64}(0x1) isa AK.CounterRNG{AK.SplitMix64, UInt64}
    @test AK.CounterRNG{AK.Philox}(UInt32(0x1)) isa AK.CounterRNG{AK.Philox, UInt32}
    @test AK.CounterRNG{AK.Threefry, UInt16}(123) isa AK.CounterRNG{AK.Threefry, UInt16}
    @test_throws ArgumentError AK.CounterRNG{AK.SplitMix64, UInt8}(300)

    rng_algs = (AK.SplitMix64(), AK.Philox(), AK.Threefry())

    for alg in rng_algs
        rng_alg = AK.CounterRNG(0x123456789abcdef; alg)
        @test AK.rand_uint32(rng_alg, UInt64(0)) == AK.rand_uint32(rng_alg, UInt64(0))
        @test AK.rand_uint32(rng_alg, UInt64(1)) != AK.rand_uint32(rng_alg, UInt64(0))

        vals_alg = [AK.rand_uint32(rng_alg, UInt64(i)) for i in 0:1023]
        @test length(unique(vals_alg)) == length(vals_alg)

        x_alg = array_from_host(zeros(Float32, 2048))
        AK.rand!(rng_alg, x_alg; prefer_threads, block_size=64)
        @test _is_unit_interval(Array(x_alg))
    end

    rng = AK.CounterRNG(0x123456789abcdef)

    @test AK.rand_uint32(rng, UInt64(0)) == AK.rand_uint32(rng, UInt64(0))
    @test AK.rand_uint32(rng, UInt64(1)) != AK.rand_uint32(rng, UInt64(0))
    @test AK.rand_uint32(rng, UInt64(17)) != AK.rand_uint32(rng, UInt64(18))

    vals = [AK.rand_uint32(rng, UInt64(i)) for i in 0:2047]
    @test length(unique(vals)) == length(vals)

    for u in (
        UInt32(0x00000000),
        UInt32(0x00000001),
        UInt32(0x7fffffff),
        UInt32(0x80000000),
        UInt32(0xffffffff),
    )
        x = AK.uint32_to_unit_float32(u)
        @test !isnan(x)
        @test 0.0f0 <= x < 1.0f0
    end

    lengths = (0, 1, 31, 32, 33, 1024, 1025)
    for len in lengths
        x = array_from_host(zeros(Float32, len))
        AK.rand!(rng, x; prefer_threads, block_size=64)
        xh = Array(x)

        ref = zeros(Float32, len)
        _rand_fill_reference!(rng, ref)

        @test xh == ref
        @test _is_unit_interval(xh)
    end

    x1 = array_from_host(zeros(Float32, 4096))
    x2 = array_from_host(zeros(Float32, 4096))
    AK.rand!(rng, x1; prefer_threads, block_size=64)
    AK.rand!(rng, x2; prefer_threads, block_size=257)
    @test Array(x1) == Array(x2)

    rng2 = AK.CounterRNG(rng.seed + UInt64(1))
    x3 = array_from_host(zeros(Float32, 4096))
    AK.rand!(rng2, x3; prefer_threads, block_size=64)
    @test Array(x3) != Array(x1)

    xnd = array_from_host(zeros(Float32, 7, 11, 5))
    AK.rand!(rng, xnd; prefer_threads, block_size=128)
    xndh = Array(xnd)
    refnd = zeros(Float32, 7, 11, 5)
    _rand_fill_reference!(rng, refnd)
    @test xndh == refnd

    if IS_CPU_BACKEND
        base = zeros(Float32, 64)
        view_x = @view base[2:2:end]
        AK.rand!(
            rng,
            view_x;
            max_tasks=Threads.nthreads(),
            min_elems=1,
            prefer_threads=true,
        )

        ref_view = zeros(Float32, length(view_x))
        _rand_fill_reference!(rng, ref_view)
        @test collect(view_x) == ref_view
    end

    nstats = 200_000
    xstats = array_from_host(zeros(Float32, nstats))
    AK.rand!(rng, xstats; prefer_threads, block_size=256)
    xh = Array(xstats)

    @test _is_unit_interval(xh)

    m = sum(xh) / nstats
    v = sum((x - m)^2 for x in xh) / nstats
    @test abs(m - 0.5) < 0.01
    @test abs(v - (1 / 12)) < 0.01

    nbins = 16
    counts = zeros(Int, nbins)
    for x in xh
        ibin = Int(floor(x * nbins)) + 1
        ibin = min(ibin, nbins)
        counts[ibin] += 1
    end
    expected = nstats / nbins
    max_rel_dev = maximum(abs(c - expected) / expected for c in counts)
    @test max_rel_dev < 0.1

    x64 = array_from_host(zeros(Float64, 16))
    @test_throws MethodError AK.rand!(rng, x64; prefer_threads)
end
