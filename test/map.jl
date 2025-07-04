@testset "map" begin
    Random.seed!(0)

    # CPU
    if BACKEND == get_backend([])
        x = Array(1:1000)
        y = AK.map(x; use_KA) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = Array(1:1000)
        y = zeros(Int, 1000)
        AK.map!(y, x; use_KA) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = rand(Float32, 1000)
        y = AK.map(x; use_KA, max_tasks=2, min_elems=100) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        x = rand(Float32, 1000)
        y = AK.map(x; use_KA, max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        # Test that undefined kwargs are not accepted
        @test_throws MethodError AK.map(x -> x^2, x; use_KA, bad=:kwarg)
    # GPU
    else
        x = array_from_host(1:1000)
        y = AK.map(x; use_KA) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(1:1000)
        y = array_from_host(zeros(Int, 1000))
        AK.map!(y, x; use_KA) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(rand(Float32, 1000))
        y = AK.map(x; use_KA, block_size=64) do i
            i > 0.5 ? i : 0
        end
        @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))

        # Test that undefined kwargs are not accepted
        @test_throws MethodError AK.map(x -> x^2, x; use_KA, bad=:kwarg)
    end
end
