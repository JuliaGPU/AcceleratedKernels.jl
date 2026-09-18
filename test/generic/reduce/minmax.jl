
@testset "minimum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.minimum(v; prefer_threads) == minimum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.minimum(v; prefer_threads) == minimum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.minimum(v; prefer_threads) == minimum(vh)

            # Along dimensions
            r = Array(AK.minimum(v; prefer_threads, dims))
            rh = minimum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.minimum(v; prefer_threads, block_size=64)
    @test AK.minimum(v; prefer_threads, dims=:) == minimum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.minimum(v; prefer_threads, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "maximum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.maximum(v; prefer_threads) == maximum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.maximum(v; prefer_threads) == maximum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.maximum(v; prefer_threads) == maximum(vh)

            # Along dimensions
            r = Array(AK.maximum(v; prefer_threads, dims))
            rh = maximum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.maximum(v; prefer_threads, block_size=64)
    @test AK.maximum(v; prefer_threads, dims=:) == maximum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.maximum(v; prefer_threads, bad=:kwarg)

    # The other settings are stress-tested in reduce
end
