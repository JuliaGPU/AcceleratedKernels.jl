function _is_unit_interval(v)
    for x in v
        if isnan(x) || x < zero(x) || x >= one(x)
            return false
        end
    end
    return true
end

function _rand_fill_reference!(rng, x::AbstractArray{T}) where {T <: AK.ALLOWED_RAND_SCALARS}
    @inbounds for i in eachindex(x)
        counter = UInt64(i - one(i))
        x[i] = AK.rand_scalar(rng, counter, T)
    end
    return x
end

@testset "rand" begin
    @test AK.CounterRNG(0x1; alg=AK.SplitMix64()) isa AK.CounterRNG{AK.SplitMix64}
    @test AK.CounterRNG(UInt32(0x1); alg=AK.Philox()) isa AK.CounterRNG{AK.Philox}
    @test AK.CounterRNG(UInt16(123); alg=AK.Threefry()) isa AK.CounterRNG{AK.Threefry}
    @test AK.CounterRNG(UInt32(300)).seed == UInt64(300)
    @test_throws ArgumentError AK.CounterRNG(-1)

    Random.seed!(0x1234)
    expected_auto_seed = rand(UInt64)
    Random.seed!(0x1234)
    rng_auto = AK.CounterRNG()
    @test rng_auto.seed == expected_auto_seed
    @test rng_auto.alg isa AK.SplitMix64

    xauto1 = array_from_host(zeros(Float32, 1024))
    xauto2 = array_from_host(zeros(Float32, 1024))
    AK.rand!(rng_auto, xauto1; prefer_threads, block_size=64)
    AK.rand!(rng_auto, xauto2; prefer_threads, block_size=257)
    @test Array(xauto1) == Array(xauto2)

    Random.seed!(0xabcdef)
    seed1 = rand(UInt64)
    seed2 = rand(UInt64)
    ref1 = array_from_host(zeros(Float32, 1024))
    ref2 = array_from_host(zeros(Float32, 1024))
    AK.rand!(AK.CounterRNG(seed1; alg=AK.SplitMix64()), ref1; prefer_threads, block_size=64)
    AK.rand!(AK.CounterRNG(seed2; alg=AK.SplitMix64()), ref2; prefer_threads, block_size=64)

    Random.seed!(0xabcdef)
    xconv1 = array_from_host(zeros(Float32, 1024))
    xconv2 = array_from_host(zeros(Float32, 1024))
    AK.rand!(xconv1; prefer_threads, block_size=64)
    AK.rand!(xconv2; prefer_threads, block_size=64)
    @test Array(xconv1) == Array(ref1)
    @test Array(xconv2) == Array(ref2)

    rng_algs = (AK.SplitMix64(), AK.Philox(), AK.Threefry())
    scalar_types = (UInt32, UInt64, Int32, Int64, Float32, Float64)

    for alg in rng_algs
        rng_alg = AK.CounterRNG(0x123456789abcdef; alg)
        for U in (UInt32, UInt64)
            @test AK.rand_uint(rng_alg, UInt64(0), U) == AK.rand_uint(rng_alg, UInt64(0), U)
            @test AK.rand_uint(rng_alg, UInt64(1), U) != AK.rand_uint(rng_alg, UInt64(0), U)

            vals_alg = [AK.rand_uint(rng_alg, UInt64(i), U) for i in 0:1023]
            @test length(unique(vals_alg)) > 900
        end

        for T in scalar_types
            x_alg = array_from_host(zeros(T, 2048))
            AK.rand!(rng_alg, x_alg; prefer_threads, block_size=64)
            x_alg_h = Array(x_alg)
            ref_alg = zeros(T, 2048)
            _rand_fill_reference!(rng_alg, ref_alg)
            @test x_alg_h == ref_alg
            if T <: AbstractFloat
                @test _is_unit_interval(x_alg_h)
            end
        end
    end

    rng = AK.CounterRNG(0x123456789abcdef)

    @test AK.from_uint(UInt32, UInt32(0xdeadbeef)) == UInt32(0xdeadbeef)
    @test AK.from_uint(UInt64, UInt64(0x0123456789abcdef)) == UInt64(0x0123456789abcdef)
    @test AK.from_uint(Int32, UInt32(0xdeadbeef)) == reinterpret(Int32, UInt32(0xdeadbeef))
    @test AK.from_uint(Int64, UInt64(0x0123456789abcdef)) == reinterpret(Int64, UInt64(0x0123456789abcdef))

    @test AK.rand_uint(rng, UInt64(0), UInt32) == AK.rand_uint(rng, UInt64(0), UInt32)
    @test AK.rand_uint(rng, UInt64(1), UInt32) != AK.rand_uint(rng, UInt64(0), UInt32)
    @test AK.rand_uint(rng, UInt64(17), UInt32) != AK.rand_uint(rng, UInt64(18), UInt32)
    @test AK.rand_uint(rng, UInt64(0), UInt64) == AK.rand_uint(rng, UInt64(0), UInt64)
    @test AK.rand_uint(rng, UInt64(1), UInt64) != AK.rand_uint(rng, UInt64(0), UInt64)
    @test AK.rand_uint(rng, UInt64(17), UInt64) != AK.rand_uint(rng, UInt64(18), UInt64)

    vals_u32 = [AK.rand_uint(rng, UInt64(i), UInt32) for i in 0:2047]
    vals_u64 = [AK.rand_uint(rng, UInt64(i), UInt64) for i in 0:2047]
    @test length(unique(vals_u32)) > 1800
    @test length(unique(vals_u64)) > 2000

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

    for u in (
        UInt64(0x0000000000000000),
        UInt64(0x0000000000000001),
        UInt64(0x7fffffffffffffff),
        UInt64(0x8000000000000000),
        UInt64(0xffffffffffffffff),
    )
        x = AK.uint64_to_unit_float64(u)
        @test !isnan(x)
        @test 0.0 <= x < 1.0
    end

    for T in scalar_types
        s0 = AK.rand_scalar(rng, UInt64(0), T)
        s1 = AK.rand_scalar(rng, UInt64(1), T)
        @test s0 isa T
        @test s1 isa T
        @test s0 != s1
        if T <: AbstractFloat
            @test zero(T) <= s0 < one(T)
            @test zero(T) <= s1 < one(T)
        end
    end

    @test_throws ArgumentError AK.rand_scalar(rng, UInt64(0), UInt16)

    lengths = (0, 1, 31, 32, 33, 1024, 1025)
    for T in scalar_types
        for len in lengths
            x = array_from_host(zeros(T, len))
            AK.rand!(rng, x; prefer_threads, block_size=64)
            xh = Array(x)

            ref = zeros(T, len)
            _rand_fill_reference!(rng, ref)

            @test xh == ref
            if T <: AbstractFloat
                @test _is_unit_interval(xh)
            end
        end
    end

    rng2 = AK.CounterRNG(rng.seed + UInt64(1))
    for T in scalar_types
        x1 = array_from_host(zeros(T, 4096))
        x2 = array_from_host(zeros(T, 4096))
        AK.rand!(rng, x1; prefer_threads, block_size=64)
        AK.rand!(rng, x2; prefer_threads, block_size=257)
        @test Array(x1) == Array(x2)

        x3 = array_from_host(zeros(T, 4096))
        AK.rand!(rng2, x3; prefer_threads, block_size=64)
        @test Array(x3) != Array(x1)

        xnd = array_from_host(zeros(T, 7, 11, 5))
        AK.rand!(rng, xnd; prefer_threads, block_size=128)
        xndh = Array(xnd)
        refnd = zeros(T, 7, 11, 5)
        _rand_fill_reference!(rng, refnd)
        @test xndh == refnd
    end

    if IS_CPU_BACKEND
        for T in scalar_types
            base = zeros(T, 64)
            view_x = @view base[2:2:end]
            AK.rand!(
                rng,
                view_x;
                max_tasks=Threads.nthreads(),
                min_elems=1,
                prefer_threads=true,
            )

            ref_view = zeros(T, length(view_x))
            _rand_fill_reference!(rng, ref_view)
            @test collect(view_x) == ref_view
        end
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

    x16 = array_from_host(zeros(UInt16, 16))
    @test_throws ArgumentError AK.rand!(x16; prefer_threads)
    @test_throws ArgumentError AK.rand!(rng, x16; prefer_threads)
end
