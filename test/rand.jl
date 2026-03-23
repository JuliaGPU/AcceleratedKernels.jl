const RAND_ALGS = (AK.SplitMix64(), AK.Philox(), AK.Threefry())
const RAND_SCALAR_TYPES_ALL = (
    UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float16, Float32, Float64,
    Bool,
)
const RAND_SCALAR_TYPES_BACKEND = IS_CPU_BACKEND ?
                                  RAND_SCALAR_TYPES_ALL :
                                  (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float32, Bool)
const RUN_FLOAT16_RAND_TESTS = IS_CPU_BACKEND
const RUN_FLOAT64_RAND_TESTS = IS_CPU_BACKEND


_is_unit_interval(v) = all(x -> !isnan(x) && zero(x) <= x < one(x), v)


function _rand_fill_reference!(
    rng,
    x::AbstractArray{T};
    counter_offset::UInt64=UInt64(0),
) where {T <: AK.ALLOWED_RAND_SCALARS}
    @inbounds for i in eachindex(x)
        x[i] = AK.rand_scalar(rng.seed, rng.alg, counter_offset + UInt64(i - one(i)), T)
    end
    return x
end


function _assert_rand_matches_reference!(rng, x; kwargs...)
    counter_offset = rng.offset
    AK.rand!(rng, x; kwargs...)
    ref = zeros(eltype(x), size(x))
    _rand_fill_reference!(rng, ref; counter_offset)
    @test Array(x) == ref
    return x
end


@testset "rand" begin
    @testset "constructors" begin
        @test AK.CounterRNG(0x1; alg=AK.SplitMix64()) isa AK.CounterRNG{AK.SplitMix64}
        @test AK.CounterRNG(UInt32(0x1); alg=AK.Philox()) isa AK.CounterRNG{AK.Philox}
        @test AK.CounterRNG(UInt16(123); alg=AK.Threefry()) isa AK.CounterRNG{AK.Threefry}
        @test AK.CounterRNG(UInt32(300)).seed == UInt64(300)
        @test AK.CounterRNG(UInt32(300)).offset == UInt64(0)
        @test AK.CounterRNG(0x1; offset=17).offset == UInt64(17)
        @test AK.CounterRNG(0x1, AK.Philox()).offset == UInt64(0)
        @test_throws ArgumentError AK.CounterRNG(-1)
        @test_throws ArgumentError AK.CounterRNG(1; offset=-1)

        Random.seed!(0x1234)
        expected_seed = Random.rand(Random.default_rng(), UInt64)
        Random.seed!(0x1234)
        rng_auto = AK.CounterRNG()
        @test rng_auto.seed == expected_seed
        @test rng_auto.alg isa AK.Philox
        @test rng_auto.offset == UInt64(0)

        rng_auto_off = AK.CounterRNG(; offset=42)
        @test rng_auto_off.offset == UInt64(42)

        x1 = array_from_host(zeros(Float32, 1024))
        x2 = array_from_host(zeros(Float32, 1024))
        AK.rand!(rng_auto, x1; prefer_threads, block_size=64)
        AK.rand!(rng_auto, x2; prefer_threads, block_size=257)
        @test rng_auto.offset == UInt64(2048)
        @test Array(x1) != Array(x2)
    end


    @testset "abstract rng offset behavior" begin
        mutable struct MutableNoOffsetRNG <: AK.AbstractCounterRNG{AK.Philox}
            seed::UInt64
            alg::AK.Philox
        end

        mutable struct MutableWithOffsetRNG <: AK.AbstractCounterRNG{AK.Philox}
            seed::UInt64
            alg::AK.Philox
            offset::UInt64
        end

        struct ImmutableWithOffsetRNG <: AK.AbstractCounterRNG{AK.Philox}
            seed::UInt64
            alg::AK.Philox
            offset::UInt64
        end

        rng_no_offset = MutableNoOffsetRNG(UInt64(0x1234), AK.Philox())
        x1 = array_from_host(zeros(Float32, 256))
        x2 = array_from_host(zeros(Float32, 256))
        AK.rand!(rng_no_offset, x1; prefer_threads, block_size=64)
        AK.rand!(rng_no_offset, x2; prefer_threads, block_size=64)
        @test Array(x1) == Array(x2)

        rng_stream = MutableWithOffsetRNG(UInt64(0x1234), AK.Philox(), UInt64(0))
        s1 = array_from_host(zeros(Float32, 100))
        s2 = array_from_host(zeros(Float32, 100))
        s12 = array_from_host(zeros(Float32, 200))
        AK.rand!(rng_stream, s1; prefer_threads, block_size=64)
        AK.rand!(rng_stream, s2; prefer_threads, block_size=64)
        AK.rand!(AK.CounterRNG(UInt64(0x1234); alg=AK.Philox()), s12; prefer_threads, block_size=64)
        @test vcat(Array(s1), Array(s2)) == Array(s12)
        @test rng_stream.offset == UInt64(200)

        rng_imm = ImmutableWithOffsetRNG(UInt64(0x1234), AK.Philox(), UInt64(17))
        y1 = array_from_host(zeros(Float32, 64))
        y2 = array_from_host(zeros(Float32, 64))
        AK.rand!(rng_imm, y1; prefer_threads, block_size=64)
        AK.rand!(rng_imm, y2; prefer_threads, block_size=64)
        @test Array(y1) == Array(y2)

        @test AK.reset!(rng_stream) === rng_stream
        @test rng_stream.offset == UInt64(0)
        @test_throws ArgumentError AK.reset!(rng_no_offset)
        @test_throws ArgumentError AK.reset!(rng_imm)
    end


    @testset "reset!" begin
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())
        x1 = array_from_host(zeros(Float32, 512))
        x2 = array_from_host(zeros(Float32, 512))

        AK.rand!(rng, x1; prefer_threads, block_size=64)
        @test rng.offset == UInt64(512)
        @test AK.reset!(rng) === rng
        @test rng.offset == UInt64(0)
        AK.rand!(rng, x2; prefer_threads, block_size=64)

        @test Array(x1) == Array(x2)
    end


    @testset "bit helpers" begin
        hi = UInt32(0b10101010101010101010101010101010)
        lo = UInt32(0b01010101010101010101010101010101)
        word = UInt64(hi) << 32 | UInt64(lo)

        @test AK._u32_hi(word) == hi
        @test AK._u32_lo(word) == lo
        @test AK._u64_from_u32s(lo, hi) == word
        @test AK._mulhi_u32(0xffffffff % UInt32, 0xffffffff % UInt32) == 0xfffffffe % UInt32
        @test AK._rotl32(0b10000000000000000000000000000001 % UInt32, UInt32(1)) == 0b11 % UInt32
        @test AK._counter_from_index(1) == UInt64(0)
        @test AK._counter_from_index(17) == UInt64(16)

        @test AK.raw_uint_type(UInt8) === UInt32
        @test AK.raw_uint_type(UInt16) === UInt32
        @test AK.raw_uint_type(UInt32) === UInt32
        @test AK.raw_uint_type(Int8) === UInt32
        @test AK.raw_uint_type(Int16) === UInt32
        @test AK.raw_uint_type(Int32) === UInt32
        if RUN_FLOAT16_RAND_TESTS
            @test AK.raw_uint_type(Float16) === UInt32
        end
        @test AK.raw_uint_type(Float32) === UInt32
        @test AK.raw_uint_type(UInt64) === UInt64
        @test AK.raw_uint_type(Int64) === UInt64
        @test AK.raw_uint_type(Bool) === UInt32
        if RUN_FLOAT64_RAND_TESTS
            @test AK.raw_uint_type(Float64) === UInt64
        end

        @test AK.from_uint(UInt8, UInt32(0xabcdef01)) == UInt8(0xab)
        @test AK.from_uint(UInt16, UInt32(0xabcdef01)) == UInt16(0xabcd)
        @test AK.from_uint(UInt32, 0b1010 % UInt32) == 0b1010 % UInt32
        @test AK.from_uint(UInt64, 0b1010 % UInt64) == 0b1010 % UInt64
        @test AK.from_uint(Int8, UInt32(0xff000000)) == Int8(-1)
        @test AK.from_uint(Int16, UInt32(0xffff0000)) == Int16(-1)
        @test AK.from_uint(Int32, 0b11111111111111111111111111111111 % UInt32) == Int32(-1)
        @test AK.from_uint(
            Int64, 0b1111111111111111111111111111111111111111111111111111111111111111 % UInt64
        ) == Int64(-1)
        if RUN_FLOAT16_RAND_TESTS
            @test AK.from_uint(Float16, UInt32(0)) == Float16(0)
        end
        @test AK.from_uint(Bool, UInt32(0)) == false
        @test AK.from_uint(Bool, UInt32(1)) == true

        if RUN_FLOAT16_RAND_TESTS
            @test AK.uint32_to_unit_float16(UInt32(0)) == Float16(0)
            @test Float16(0) <= AK.uint32_to_unit_float16(typemax(UInt32)) < Float16(1)
        end
        @test AK.uint32_to_unit_float32(UInt32(0)) == 0.0f0
        @test 0.0f0 <= AK.uint32_to_unit_float32(typemax(UInt32)) < 1.0f0
        if RUN_FLOAT64_RAND_TESTS
            @test AK.uint64_to_unit_float64(UInt64(0)) == 0.0
            @test 0.0 <= AK.uint64_to_unit_float64(typemax(UInt64)) < 1.0
        end
    end


    @testset "rand_uint" begin
        for alg in RAND_ALGS
            rng = AK.CounterRNG(0x123456789abcdef; alg)
            for U in (UInt32, UInt64)
                @test AK.rand_uint(rng.seed, rng.alg, UInt64(0), U) == AK.rand_uint(rng.seed, rng.alg, UInt64(0), U)
                @test AK.rand_uint(rng.seed, rng.alg, UInt64(1), U) != AK.rand_uint(rng.seed, rng.alg, UInt64(0), U)

                vals = [AK.rand_uint(rng.seed, rng.alg, UInt64(i), U) for i in 0:511]
                @test length(unique(vals)) > 460
            end
        end

        rng_splitmix = AK.CounterRNG(0x31415926; alg=AK.SplitMix64())
        for c in (UInt64(0), UInt64(1), UInt64(17), UInt64(1023))
            @test AK.rand_uint(rng_splitmix.seed, rng_splitmix.alg, c, UInt32) == AK._u32_hi(
                AK.rand_uint(rng_splitmix.seed, rng_splitmix.alg, c, UInt64)
            )
        end

        for alg in (AK.Philox(), AK.Threefry())
            rng = AK.CounterRNG(0xabcdef1234567890; alg)
            for c in (UInt64(0), UInt64(1), UInt64(17), UInt64(1023))
                @test AK._u32_lo(AK.rand_uint(rng.seed, rng.alg, c, UInt64)) ==
                      AK.rand_uint(rng.seed, rng.alg, c, UInt32)
            end
        end
    end


    @testset "rand_scalar" begin
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())

        for T in RAND_SCALAR_TYPES_BACKEND
            s0 = AK.rand_scalar(rng.seed, rng.alg, UInt64(0), T)
            s1 = AK.rand_scalar(rng.seed, rng.alg, UInt64(1), T)
            @test s0 isa T
            @test s1 isa T
            @test s0 == AK.rand_scalar(rng.seed, rng.alg, UInt64(0), T)
            if !(T in (Bool, Float16, UInt8, UInt16, Int8, Int16))
                @test s0 != s1
            end
            if T <: AbstractFloat
                @test zero(T) <= s0 < one(T)
                @test zero(T) <= s1 < one(T)
            end
        end

        c = UInt64(42)
        @test AK.rand_scalar(rng.seed, rng.alg, c, UInt8) ==
              trunc(UInt8, AK.rand_uint(rng.seed, rng.alg, c, UInt32) >> 24)
        @test AK.rand_scalar(rng.seed, rng.alg, c, UInt16) ==
              trunc(UInt16, AK.rand_uint(rng.seed, rng.alg, c, UInt32) >> 16)
        @test AK.rand_scalar(rng.seed, rng.alg, c, Int8) ==
              reinterpret(Int8, trunc(UInt8, AK.rand_uint(rng.seed, rng.alg, c, UInt32) >> 24))
        @test AK.rand_scalar(rng.seed, rng.alg, c, Int16) ==
              reinterpret(Int16, trunc(UInt16, AK.rand_uint(rng.seed, rng.alg, c, UInt32) >> 16))
        @test AK.rand_scalar(rng.seed, rng.alg, c, Int32) ==
              reinterpret(Int32, AK.rand_uint(rng.seed, rng.alg, c, UInt32))
        @test AK.rand_scalar(rng.seed, rng.alg, c, Int64) ==
              reinterpret(Int64, AK.rand_uint(rng.seed, rng.alg, c, UInt64))
        if RUN_FLOAT16_RAND_TESTS
            @test AK.rand_scalar(rng.seed, rng.alg, c, Float16) == AK.uint32_to_unit_float16(
                AK.rand_uint(rng.seed, rng.alg, c, UInt32)
            )
        end
        @test AK.rand_scalar(rng.seed, rng.alg, c, Float32) == AK.uint32_to_unit_float32(
            AK.rand_uint(rng.seed, rng.alg, c, UInt32)
        )
        @test AK.rand_scalar(rng.seed, rng.alg, c, Bool) ==
              isodd(AK.rand_uint(rng.seed, rng.alg, c, UInt32))
        if RUN_FLOAT64_RAND_TESTS
            @test AK.rand_scalar(rng.seed, rng.alg, c, Float64) == AK.uint64_to_unit_float64(
                AK.rand_uint(rng.seed, rng.alg, c, UInt64)
            )
        end
        bools = [AK.rand_scalar(rng.seed, rng.alg, UInt64(i), Bool) for i in 0:511]
        @test any(identity, bools)
        @test any(!, bools)
        @test_throws ArgumentError AK.rand_scalar(rng.seed, rng.alg, UInt64(0), UInt128)
    end


    @testset "rand! explicit rng" begin
        lengths = (0, 1, 31, 32, 33, 257, 1024)
        rng = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())

        for T in RAND_SCALAR_TYPES_BACKEND
            for len in lengths
                x = array_from_host(zeros(T, len))
                _assert_rand_matches_reference!(rng, x; prefer_threads, block_size=64)
                if T <: AbstractFloat
                    @test _is_unit_interval(Array(x))
                end
            end
        end

        for T in RAND_SCALAR_TYPES_BACKEND
            x1 = array_from_host(zeros(T, 2048))
            x2 = array_from_host(zeros(T, 2048))
            rng1 = AK.CounterRNG(rng.seed; alg=rng.alg)
            rng2 = AK.CounterRNG(rng.seed; alg=rng.alg)
            AK.rand!(rng1, x1; prefer_threads, block_size=64)
            AK.rand!(rng2, x2; prefer_threads, block_size=257)
            @test Array(x1) == Array(x2)
        end

        for T in RAND_SCALAR_TYPES_BACKEND
            rng1 = AK.CounterRNG(rng.seed; alg=rng.alg)
            rng2 = AK.CounterRNG(rng.seed + UInt64(1); alg=rng.alg)
            x1 = array_from_host(zeros(T, 2048))
            x2 = array_from_host(zeros(T, 2048))
            AK.rand!(rng1, x1; prefer_threads, block_size=64)
            AK.rand!(rng2, x2; prefer_threads, block_size=64)
            @test Array(x1) != Array(x2)
        end

        begin
            rng_stream = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())
            rng_once = AK.CounterRNG(0x123456789abcdef; alg=AK.Philox())
            x1 = array_from_host(zeros(Float32, 100))
            x2 = array_from_host(zeros(Float32, 100))
            x12 = array_from_host(zeros(Float32, 200))
            AK.rand!(rng_stream, x1; prefer_threads, block_size=64)
            AK.rand!(rng_stream, x2; prefer_threads, block_size=64)
            AK.rand!(rng_once, x12; prefer_threads, block_size=64)
            @test vcat(Array(x1), Array(x2)) == Array(x12)
            @test rng_stream.offset == UInt64(200)
        end

        for T in (RUN_FLOAT16_RAND_TESTS ? (Float16, Float32, UInt64, Bool) : (Float32, UInt64, Bool))
            xnd = array_from_host(zeros(T, 7, 11, 5))
            _assert_rand_matches_reference!(rng, xnd; prefer_threads, block_size=128)
        end

        if IS_CPU_BACKEND
            for T in RAND_SCALAR_TYPES_BACKEND
                base = zeros(T, 64)
                view_x = @view base[2:2:end]
                AK.rand!(
                    rng, view_x;
                    max_tasks=Threads.nthreads(),
                    min_elems=1,
                    prefer_threads=true
                )
                ref_view = zeros(T, length(view_x))
                _rand_fill_reference!(
                    rng, ref_view;
                    counter_offset=rng.offset - UInt64(length(view_x)),
                )
                @test collect(view_x) == ref_view
            end
        end
    end


    @testset "rand! convenience" begin
        ref1 = array_from_host(zeros(Float32, 1024))
        ref2 = array_from_host(zeros(Float32, 1024))
        x1 = array_from_host(zeros(Float32, 1024))
        x2 = array_from_host(zeros(Float32, 1024))

        Random.seed!(0xabcdef)
        seed1 = Random.rand(Random.default_rng(), UInt64)
        AK.rand!(AK.CounterRNG(seed1; alg=AK.Philox()), ref1; prefer_threads, block_size=64)
        seed2 = Random.rand(Random.default_rng(), UInt64)
        AK.rand!(AK.CounterRNG(seed2; alg=AK.Philox()), ref2; prefer_threads, block_size=64)

        Random.seed!(0xabcdef)
        AK.rand!(x1; prefer_threads, block_size=64)
        AK.rand!(x2; prefer_threads, block_size=64)
        @test Array(x1) == Array(ref1)
        @test Array(x2) == Array(ref2)

        x_bad = zeros(UInt128, 16)
        @test_throws ArgumentError AK.rand!(x_bad; prefer_threads)
        @test_throws ArgumentError AK.rand!(AK.CounterRNG(0x1), x_bad; prefer_threads)
    end
end
