ALGS = AK.AccumulateAlgorithm[AK.ScanPrefixes()]

@isdefined(TEST_DL) && TEST_DL[] && push!(ALGS, AK.DecoupledLookback())

@testset "accumulate_1d $(alg isa AK.DecoupledLookback ? "DL" : "SP")" for alg in ALGS

    Random.seed!(0)

    # Single block exlusive scan (each block processes two elements)
    for num_elems in 1:256
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; prefer_threads, init=0, inclusive=false, block_size=128, alg)
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Single block inclusive scan
    for num_elems in 1:256
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; prefer_threads, init=0, block_size=128, alg)
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Large exclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; prefer_threads, init=0, inclusive=false, alg)
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Large inclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; prefer_threads, init=0, alg)
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Stress-testing small block sizes -> many blocks
    for _ in 1:100
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; prefer_threads, init=0, block_size=16, alg)
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)
        vh = rand(Float32, n1, n2, n3)
        v = array_from_host(vh)
        AK.accumulate!(+, v; prefer_threads, init=0, alg)
        @test all(Array(v) .≈ accumulate(+, vh))
    end

    # Ensuring the init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = similar(x)
        init = rand(-1000:1000)
        AK.accumulate!(+, y, x; prefer_threads, init=Int32(init), alg)
        @test all(Array(y) .== accumulate(+, Array(x); init))
    end

    # Exclusive scan
    x = array_from_host(ones(Int32, 10))
    y = copy(x)
    AK.accumulate!(+, y; prefer_threads, init=0, inclusive=false, alg)
    @test all(Array(y) .== 0:9)

    # Test init value is respected with exclusive scan too
    x = array_from_host(ones(Int32, 10))
    y = copy(x)
    init = 10
    AK.accumulate!(+, y; prefer_threads, init=Int32(init), inclusive=false, alg)
    @test all(Array(y) .== 10:19)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.accumulate(+, y; prefer_threads, init=10, dims=2, inclusive=false, bad=:kwarg)

    # Testing different settings
    AK.accumulate!(+, array_from_host(ones(Int32, 1000)); init=0, inclusive=false,
                prefer_threads, block_size=128, alg,
                temp=array_from_host(zeros(Int32, 1000)),
                temp_flags=array_from_host(zeros(Int8, 1000)))
    AK.accumulate(+, array_from_host(ones(Int32, 1000)); init=0, inclusive=false,
                prefer_threads, block_size=128, alg,
                temp=array_from_host(zeros(Int64, 1000)),
                temp_flags=array_from_host(zeros(Int8, 1000)))
end


@testset "accumulate_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.accumulate
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.accumulate(+, s; prefer_threads, init=Int32(0), dims)

                    dh = Array(d)
                    dhres = accumulate(+, sh; init=Int32(0), dims)
                    @test dh == dhres
                    @test eltype(dh) == eltype(dhres)
                end
            end
        end
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)

            s = AK.accumulate(+, v; prefer_threads, init=Int32(0), dims)
            sh = Array(s)
            @test sh == accumulate(+, vh; init=Int32(0), dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)

            s = AK.accumulate(+, v; prefer_threads, init=UInt32(0), dims)
            sh = Array(s)
            @test sh == accumulate(+, vh; init=UInt32(0), dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)

            s = AK.accumulate(+, v; prefer_threads, init=Float32(0), dims)
            sh = Array(s)
            @test all(sh .≈ accumulate(+, vh; init=Float32(0), dims))
        end
    end

    # Ensure the init value is respected
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            init = rand(-1000:1000)
            s = AK.accumulate(+, v; prefer_threads, init=Float32(init), dims)
            sh = Array(s)
            @test all(sh .≈ accumulate(+, vh; init=Float32(init), dims))
        end
    end

    # Exclusive scan
    vh = ones(Int32, 10, 10)
    v = array_from_host(vh)
    s = AK.accumulate(+, v; prefer_threads, init=0, dims=2, inclusive=false)
    sh = Array(s)
    @test all([sh[i, :] == 0:9 for i in 1:10])

    # Test init value is respected with exclusive scan too
    vh = ones(Int32, 10, 10)
    v = array_from_host(vh)
    s = AK.accumulate(+, v; prefer_threads, init=10, dims=2, inclusive=false)
    sh = Array(s)
    @test all([sh[i, :] == 10:19 for i in 1:10])

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.accumulate(+, v; prefer_threads, init=10, dims=2, inclusive=false, bad=:kwarg)

    # Test all options with bigger matrices
    for D in [(1_000_000,3), (3,1_000_000)], dims in [1,2]
        @testset let D = D, dims = dims
            vh = ones(Float32, D)
            v = array_from_host(vh)
            s = AK.accumulate(+, v; init=0, dims)
            sh = Array(s)
            @test sh == accumulate(+, vh; init=0, dims)
        end
    end

    # Testing different settings
    AK.accumulate(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        init=Int32(0),
        neutral=Int32(0),
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 1, 5)),
    )
    AK.accumulate(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        init=Int32(0),
        neutral=Int32(0),
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 4, 1)),
    )
end
@testset "cumsum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    vh = Array(v)
    @test Array(AK.cumsum(v; prefer_threads)) == cumsum(vh)

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        vh = rand(Float32, num_elems)
        v = array_from_host(vh)
        @test all(Array(AK.cumsum(v; prefer_threads)) .≈ cumsum(vh))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear; not supported in Base
            # @test all(Array(AK.cumsum(v; prefer_threads)) .== cumsum(vh))

            # Along dimensions
            r = Array(AK.cumsum(v; prefer_threads, dims))
            rh = cumsum(vh; dims)

            @test r == rh
        end
    end

    # Test promotion to op-dictated type
    xh = rand(Bool, 16)
    x = array_from_host(xh)
    @test Array(AK.cumsum(x; prefer_threads)) == cumsum(xh)

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.cumsum(v; prefer_threads, block_size=64)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.cumsum(v; prefer_threads, init=10, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "cumprod" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    vh = Array(v)
    @test Array(AK.cumprod(v; prefer_threads)) == cumprod(vh)

    vh = ones(Float32, 100_000)
    v = array_from_host(vh)
    @test Array(AK.cumprod(v; prefer_threads)) == vh

    # Fuzzy testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear; not supported in Base
            # @test all(Array(AK.cumprod(v; prefer_threads)) .== cumprod(vh))

            # Along dimensions
            r = Array(AK.cumprod(v; prefer_threads, dims))
            rh = cumprod(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.cumprod(v; prefer_threads, block_size=64)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.cumprod(v; prefer_threads, init=10, bad=:kwarg)

    # The other settings are stress-tested in reduce
end
