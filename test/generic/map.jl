@testset "map" begin
    Random.seed!(0)

    # CPU
    if IS_CPU_BACKEND && prefer_threads
        x = Array(1:1000)
        y = AK.map(x; prefer_threads) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = Array(1:1000)
        y = zeros(Int, 1000)
        AK.map!(y, x; prefer_threads) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = rand(Float32, 1000)
        y = AK.map(x; prefer_threads, max_tasks=2, min_elems=100) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        x = rand(Float32, 1000)
        y = AK.map(x; prefer_threads, max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        # Test that undefined kwargs are not accepted
        @test_throws MethodError AK.map(x -> x^2, x; prefer_threads, bad=:kwarg)
    # GPU
    else
        x = array_from_host(1:1000)
        y = AK.map(x; prefer_threads) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(1:1000)
        y = array_from_host(zeros(Int, 1000))
        AK.map!(y, x; prefer_threads) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(rand(Float32, 1000))
        y = AK.map(x; prefer_threads, block_size=64) do i
            i > 0.5 ? i : 0
        end
        @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))

        # Test that undefined kwargs are not accepted
        @test_throws MethodError AK.map(x -> x^2, x; prefer_threads, bad=:kwarg)
    end
end
