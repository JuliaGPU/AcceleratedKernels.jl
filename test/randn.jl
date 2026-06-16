const RANDN_ALGS = (AK.SplitMix64(), AK.Philox(), AK.Threefry())
const RANDN_FLOAT_TYPES_BACKEND = IS_CPU_BACKEND ? (Float16, Float32, Float64) : (Float32,)
const RANDN_LENGTHS = (0, 1, 2, 31, 32, 33, 257, 1024)


_all_finite(v) = all(isfinite, v)
_randn_reference_atol(::Type{Float16}) = 16 * eps(Float16)
_randn_reference_atol(::Type{Float32}) = 64 * eps(Float32)
_randn_reference_atol(::Type{Float64}) = 64 * eps(Float64)


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
    xa = Array(x)

    if IS_CPU_BACKEND
        @test xa == ref
    else
        # randn uses Box-Muller (`log`, `sqrt`, `sincos`), and GPU libm implementations are not
        # bit-identical to CPU scalar libm. Stream/counter mapping is still deterministic, but the
        # final Float32 values can differ by a few ULP, so we use a tight absolute tolerance here.
        atol = _randn_reference_atol(eltype(xa))
        @test all(isapprox.(xa, ref; rtol=zero(atol), atol))
    end

    return x
end


@testset "randn" begin
    @testset "scalar helpers" begin
        @test 0.0f0 < AK._uint32_to_open_unit_float32_midpoint(UInt32(0)) < 1.0f0
        @test 0.0f0 < AK._uint32_to_open_unit_float32_midpoint(typemax(UInt32)) < 1.0f0

        if IS_CPU_BACKEND
            @test 0.0 < AK._uint64_to_open_unit_float64_midpoint(UInt64(0)) < 1.0
            @test 0.0 < AK._uint64_to_open_unit_float64_midpoint(typemax(UInt64)) < 1.0
        end

        seed = UInt64(0x123456789abcdef)
        for alg in RANDN_ALGS
            for counter in (UInt64(0), UInt64(1), UInt64(17), UInt64(1023))
                u32 = AK.rand_float_open01(seed, alg, counter, Float32)
                @test 0.0f0 < u32 < 1.0f0

                if IS_CPU_BACKEND
                    u64 = AK.rand_float_open01(seed, alg, counter, Float64)
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
                @test p0 == AK.randn_scalar(seed, alg, UInt64(42), T)
                @test p1 == AK.randn_scalar(seed, alg, UInt64(43), T)
            end
        end

        @test_throws ArgumentError AK.rand_float_open01(seed, AK.Philox(), UInt64(0), UInt32)
        @test_throws ArgumentError AK.randn_scalar(seed, AK.Philox(), UInt64(0), UInt32)
    end


    @testset "randn! explicit rng" begin
        for alg in RANDN_ALGS
            rng = AK.CounterRNG(0x123456789abcdef; alg)

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
                x1 = array_from_host(zeros(T, 2048))
                x2 = array_from_host(zeros(T, 2048))
                rng1 = AK.CounterRNG(rng.seed; alg=rng.alg)
                rng2 = AK.CounterRNG(rng.seed + UInt64(1); alg=rng.alg)

                AK.randn!(rng1, x1; prefer_threads, block_size=64)
                AK.randn!(rng2, x2; prefer_threads, block_size=64)
                @test Array(x1) != Array(x2)
            end
        end
    end


    @testset "offset and reset semantics" begin
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


    @testset "shapes and views" begin
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
                    prefer_threads=true,
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


    @testset "randn allocation convenience" begin
        default_alloc_type = IS_CPU_BACKEND ? Float64 : Float32

        rng = AK.CounterRNG(UInt64(0x1234); alg=AK.Philox())
        y = AK.randn(rng, BACKEND, Float32, Int32(6), UInt16(7); prefer_threads, block_size=64)
        @test size(y) == (6, 7)
        @test eltype(y) === Float32
        @test _all_finite(Array(y))
        @test rng.offset == UInt64(length(y))

        rng_default = AK.CounterRNG(UInt64(0x99); alg=AK.Philox())
        rng_default_ref = AK.CounterRNG(UInt64(0x99); alg=AK.Philox())
        y_default = AK.randn(rng_default, BACKEND, 128; prefer_threads, block_size=64)
        y_default_ref = AK.randn(
            rng_default_ref, BACKEND, default_alloc_type, 128; prefer_threads, block_size=64
        )
        @test eltype(y_default) === default_alloc_type
        @test Array(y_default) == Array(y_default_ref)
        @test rng_default.offset == rng_default_ref.offset == UInt64(128)

        rng_alloc = AK.CounterRNG(UInt64(0x55); alg=AK.Philox())
        rng_fill = AK.CounterRNG(UInt64(0x55); alg=AK.Philox())
        y_alloc = AK.randn(rng_alloc, BACKEND, Float32, 128; prefer_threads, block_size=64)
        y_fill = array_from_host(zeros(Float32, 128))
        AK.randn!(rng_fill, y_fill; prefer_threads, block_size=64)
        @test Array(y_alloc) == Array(y_fill)
        @test rng_alloc.offset == rng_fill.offset == UInt64(128)

        rng_cpu_default = AK.CounterRNG(UInt64(0x66); alg=AK.Philox())
        rng_cpu_default_ref = AK.CounterRNG(UInt64(0x66); alg=AK.Philox())
        y_cpu_default = AK.randn(rng_cpu_default, 128; prefer_threads, block_size=64)
        y_cpu_default_ref = AK.randn(
            rng_cpu_default_ref, AK.get_backend([]), 128; prefer_threads, block_size=64
        )
        @test eltype(y_cpu_default) === Float64
        @test Array(y_cpu_default) == Array(y_cpu_default_ref)
        @test rng_cpu_default.offset == rng_cpu_default_ref.offset == UInt64(128)

        rng_cpu_typed = AK.CounterRNG(UInt64(0x77); alg=AK.Philox())
        rng_cpu_typed_ref = AK.CounterRNG(UInt64(0x77); alg=AK.Philox())
        y_cpu_typed = AK.randn(rng_cpu_typed, Float32, 128; prefer_threads, block_size=64)
        y_cpu_typed_ref = AK.randn(
            rng_cpu_typed_ref, AK.get_backend([]), Float32, 128; prefer_threads, block_size=64
        )
        @test eltype(y_cpu_typed) === Float32
        @test Array(y_cpu_typed) == Array(y_cpu_typed_ref)
        @test rng_cpu_typed.offset == rng_cpu_typed_ref.offset == UInt64(128)

        # Warm-up first call path so one-time compilation/backend init does not perturb RNG checks.
        AK.randn(BACKEND, Float32, 1; prefer_threads, block_size=64)

        # Auto-seeded constructor should match explicit seed capture from default RNG.
        Random.seed!(0x9abc)
        seed = Random.rand(Random.default_rng(), UInt64)
        ref = AK.randn(AK.CounterRNG(
            seed; alg=AK.Philox()), BACKEND, Float32, 64; prefer_threads, block_size=64
        )
        Random.seed!(0x9abc)
        x = AK.randn(BACKEND, Float32, 64; prefer_threads, block_size=64)
        @test Array(x) == Array(ref)

        # Auto-seeded convenience without explicit type should use backend-dependent default type.
        Random.seed!(0x4242)
        seed_default = Random.rand(Random.default_rng(), UInt64)
        ref_default = AK.randn(
            AK.CounterRNG(seed_default; alg=AK.Philox()),
            BACKEND,
            default_alloc_type,
            64;
            prefer_threads,
            block_size=64,
        )
        Random.seed!(0x4242)
        x_default = AK.randn(BACKEND, 64; prefer_threads, block_size=64)
        @test eltype(x_default) === default_alloc_type
        @test Array(x_default) == Array(ref_default)

        # Convenience without backend should default to CPU backend and Float64.
        Random.seed!(0x4545)
        seed_cpu_default = Random.rand(Random.default_rng(), UInt64)
        ref_cpu_default = AK.randn(
            AK.CounterRNG(seed_cpu_default; alg=AK.Philox()),
            AK.get_backend([]),
            Float64,
            64;
            prefer_threads,
            block_size=64,
        )
        Random.seed!(0x4545)
        x_cpu_default = AK.randn(64; prefer_threads, block_size=64)
        @test eltype(x_cpu_default) === Float64
        @test Array(x_cpu_default) == Array(ref_cpu_default)

        # Type-only convenience should default to CPU backend.
        Random.seed!(0x5656)
        seed_cpu_typed = Random.rand(Random.default_rng(), UInt64)
        ref_cpu_typed = AK.randn(
            AK.CounterRNG(seed_cpu_typed; alg=AK.Philox()),
            AK.get_backend([]),
            Float32,
            64;
            prefer_threads,
            block_size=64,
        )
        Random.seed!(0x5656)
        x_cpu_typed_no_rng = AK.randn(Float32, 64; prefer_threads, block_size=64)
        @test eltype(x_cpu_typed_no_rng) === Float32
        @test Array(x_cpu_typed_no_rng) == Array(ref_cpu_typed)

        # Reseeding should reproduce the same auto-seeded draw.
        Random.seed!(0x7777)
        x1 = AK.randn(BACKEND, Float32, 64; prefer_threads, block_size=64)
        Random.seed!(0x7777)
        x2 = AK.randn(BACKEND, Float32, 64; prefer_threads, block_size=64)
        @test Array(x1) == Array(x2)

        @test_throws ArgumentError AK.randn(AK.CounterRNG(0x1), BACKEND, UInt32, 16; prefer_threads)
        @test_throws MethodError AK.randn(
            AK.CounterRNG(0x1), BACKEND, Float32, 16; prefer_threads, bad=:kwarg
        )
        @test_throws MethodError AK.randn(BACKEND, Float32, 16; prefer_threads, bad=:kwarg)
        @test_throws MethodError AK.randn(BACKEND, 16; prefer_threads, bad=:kwarg)
        @test_throws MethodError AK.randn(16; prefer_threads, bad=:kwarg)
        @test_throws ArgumentError AK.randn()
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
