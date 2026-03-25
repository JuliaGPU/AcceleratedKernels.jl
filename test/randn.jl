const RANDN_ALGS = (AK.SplitMix64(), AK.Philox(), AK.Threefry())
const RANDN_FLOAT_TYPES_BACKEND = IS_CPU_BACKEND ? (Float16, Float32, Float64) : (Float32,)


_is_finite(v) = all(isfinite, v)


function _randn_fill_reference!(
    rng,
    x::AbstractArray{T};
    counter_offset::UInt64=UInt64(0),
) where {T <: AK.ALLOWED_RANDN_SCALARS}
    @inbounds for i in eachindex(x)
        x[i] = AK.randn_scalar(rng.seed, rng.alg, counter_offset + UInt64(i - one(i)), T)
    end
    return x
end


function _assert_randn_matches_reference!(rng, x; kwargs...)
    counter_offset = rng.offset
    AK.randn!(rng, x; kwargs...)
    ref = zeros(eltype(x), size(x))
    _randn_fill_reference!(rng, ref; counter_offset)
    @test Array(x) == ref
    return x
end


@testset "randn" begin
    @testset "open interval helpers" begin
        @test 0.0f0 < AK.uint32_to_open_unit_float32_midpoint(UInt32(0)) < 1.0f0
        @test 0.0f0 < AK.uint32_to_open_unit_float32_midpoint(typemax(UInt32)) < 1.0f0
        @test 0.0 < AK.uint64_to_open_unit_float64_midpoint(UInt64(0)) < 1.0
        @test 0.0 < AK.uint64_to_open_unit_float64_midpoint(typemax(UInt64)) < 1.0
    end


    @testset "rand_open01 and randn_scalar" begin
        seed = UInt64(0x123456789abcdef)
        for alg in RANDN_ALGS
            for c in (UInt64(0), UInt64(1), UInt64(17), UInt64(1023))
                u32 = AK.rand_open01(seed, alg, c, Float32)
                @test 0.0f0 < u32 < 1.0f0
                if IS_CPU_BACKEND
                    u64 = AK.rand_open01(seed, alg, c, Float64)
                    @test 0.0 < u64 < 1.0
                end
            end

            for T in RANDN_FLOAT_TYPES_BACKEND
                s0 = AK.randn_scalar(seed, alg, UInt64(42), T)
                s1 = AK.randn_scalar(seed, alg, UInt64(43), T)
                @test s0 isa T
                @test s1 isa T
                @test isfinite(s0)
                @test isfinite(s1)
                @test s0 == AK.randn_scalar(seed, alg, UInt64(42), T)
                @test s1 == AK.randn_scalar(seed, alg, UInt64(43), T)

                p0, p1 = AK.randn_pair(seed, alg, UInt64(21), T)
                @test AK.randn_scalar(seed, alg, UInt64(42), T) == p0
                @test AK.randn_scalar(seed, alg, UInt64(43), T) == p1
            end
        end

        @test_throws ArgumentError AK.randn_scalar(seed, AK.Philox(), UInt64(0), UInt32)
    end


    @testset "randn! explicit rng" begin
        lengths = (0, 1, 31, 32, 33, 257, 1024)

        for alg in RANDN_ALGS
            rng = AK.CounterRNG(0x123456789abcdef; alg)

            for T in RANDN_FLOAT_TYPES_BACKEND
                for len in lengths
                    x = array_from_host(zeros(T, len))
                    _assert_randn_matches_reference!(rng, x; prefer_threads, block_size=64)
                    @test _is_finite(Array(x))
                end
            end

            for T in RANDN_FLOAT_TYPES_BACKEND
                x1 = array_from_host(zeros(T, 2048))
                x2 = array_from_host(zeros(T, 2048))
                rng1 = AK.CounterRNG(rng.seed; alg=rng.alg)
                rng2 = AK.CounterRNG(rng.seed; alg=rng.alg)
                AK.randn!(rng1, x1; prefer_threads, block_size=64)
                AK.randn!(rng2, x2; prefer_threads, block_size=257)
                @test Array(x1) == Array(x2)
            end

            for T in RANDN_FLOAT_TYPES_BACKEND
                rng1 = AK.CounterRNG(rng.seed; alg=rng.alg)
                rng2 = AK.CounterRNG(rng.seed + UInt64(1); alg=rng.alg)
                x1 = array_from_host(zeros(T, 2048))
                x2 = array_from_host(zeros(T, 2048))
                AK.randn!(rng1, x1; prefer_threads, block_size=64)
                AK.randn!(rng2, x2; prefer_threads, block_size=64)
                @test Array(x1) != Array(x2)
            end
        end
    end


    @testset "counter rng offset behavior" begin
        rng_stream = AK.CounterRNG(UInt64(0x1234); alg=AK.Philox(), offset=UInt64(17))
        s1 = array_from_host(zeros(Float32, 99))
        s2 = array_from_host(zeros(Float32, 101))
        s12 = array_from_host(zeros(Float32, 200))
        AK.randn!(rng_stream, s1; prefer_threads, block_size=64)
        @test rng_stream.offset == UInt64(116)
        AK.randn!(rng_stream, s2; prefer_threads, block_size=64)
        @test rng_stream.offset == UInt64(217)

        rng_once = AK.CounterRNG(UInt64(0x1234); alg=AK.Philox(), offset=UInt64(17))
        AK.randn!(rng_once, s12; prefer_threads, block_size=64)
        @test vcat(Array(s1), Array(s2)) == Array(s12)
        @test rng_once.offset == UInt64(217)

        empty = array_from_host(zeros(Float32, 0))
        stream_offset = rng_stream.offset
        AK.randn!(rng_stream, empty; prefer_threads, block_size=64)
        @test rng_stream.offset == stream_offset

        @test AK.reset!(rng_stream) === rng_stream
        @test rng_stream.offset == UInt64(0)

        y1 = array_from_host(zeros(Float32, 64))
        y2 = array_from_host(zeros(Float32, 64))
        AK.randn!(rng_stream, y1; prefer_threads, block_size=64)
        AK.randn!(AK.CounterRNG(UInt64(0x1234); alg=AK.Philox()), y2; prefer_threads, block_size=64)
        @test Array(y1) == Array(y2)
    end


    @testset "reset!" begin
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())
        x1 = array_from_host(zeros(Float32, 512))
        x2 = array_from_host(zeros(Float32, 512))

        AK.randn!(rng, x1; prefer_threads, block_size=64)
        @test rng.offset == UInt64(512)
        @test AK.reset!(rng) === rng
        @test rng.offset == UInt64(0)
        AK.randn!(rng, x2; prefer_threads, block_size=64)

        @test Array(x1) == Array(x2)
    end


    @testset "randn! n-dimensional and views" begin
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())

        for T in RANDN_FLOAT_TYPES_BACKEND
            xnd = array_from_host(zeros(T, 7, 11, 5))
            _assert_randn_matches_reference!(rng, xnd; prefer_threads, block_size=128)
        end

        if IS_CPU_BACKEND
            for T in RANDN_FLOAT_TYPES_BACKEND
                base = zeros(T, 64)
                view_x = @view base[2:2:end]
                AK.randn!(
                    rng, view_x;
                    max_tasks=Threads.nthreads(),
                    min_elems=1,
                    prefer_threads=true
                )
                ref_view = zeros(T, length(view_x))
                _randn_fill_reference!(
                    rng, ref_view;
                    counter_offset=rng.offset - UInt64(length(view_x)),
                )
                @test collect(view_x) == ref_view
            end
        end
    end


    @testset "randn! convenience" begin
        ref1 = array_from_host(zeros(Float32, 1024))
        ref2 = array_from_host(zeros(Float32, 1024))
        x1 = array_from_host(zeros(Float32, 1024))
        x2 = array_from_host(zeros(Float32, 1024))

        Random.seed!(0xabcdef)
        seed1 = Random.rand(Random.default_rng(), UInt64)
        AK.randn!(AK.CounterRNG(seed1; alg=AK.Philox()), ref1; prefer_threads, block_size=64)
        seed2 = Random.rand(Random.default_rng(), UInt64)
        AK.randn!(AK.CounterRNG(seed2; alg=AK.Philox()), ref2; prefer_threads, block_size=64)

        Random.seed!(0xabcdef)
        AK.randn!(x1; prefer_threads, block_size=64)
        AK.randn!(x2; prefer_threads, block_size=64)
        @test Array(x1) == Array(ref1)
        @test Array(x2) == Array(ref2)

        x_bad = zeros(UInt32, 16)
        @test_throws ArgumentError AK.randn!(x_bad; prefer_threads)
        @test_throws ArgumentError AK.randn!(AK.CounterRNG(0x1), x_bad; prefer_threads)
    end


    @testset "moments sanity" begin
        n = 200_000
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())

        for T in RANDN_FLOAT_TYPES_BACKEND
            x = array_from_host(zeros(T, n))
            AK.randn!(rng, x; prefer_threads, block_size=128)
            xa = Float64.(Array(x))

            m = sum(xa) / length(xa)
            v = sum((xi - m)^2 for xi in xa) / length(xa)

            if T === Float16
                @test abs(m) < 0.1
                @test abs(v - one(v)) < 0.15
            else
                @test abs(m) < 0.01
                @test abs(v - one(v)) < 0.03
            end
        end
    end
end
