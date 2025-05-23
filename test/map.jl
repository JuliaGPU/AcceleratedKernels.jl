@testset "map" begin
    Random.seed!(0)

    # CPU
    if BACKEND == CPU()
        x = Array(1:1000)
        y = AK.map(x) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = Array(1:1000)
        y = zeros(Int, 1000)
        AK.map!(y, x) do i
            i^2
        end
        @test y == map(i -> i^2, x)

        x = rand(Float32, 1000)
        y = AK.map(x, max_tasks=2, min_elems=100) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

        x = rand(Float32, 1000)
        y = AK.map(x, max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)

    # GPU
    else
        x = array_from_host(1:1000)
        y = AK.map(x) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(1:1000)
        y = array_from_host(zeros(Int, 1000))
        AK.map!(y, x) do i
            i^2
        end
        @test Array(y) == map(i -> i^2, 1:1000)

        x = array_from_host(rand(Float32, 1000))
        y = AK.map(x, block_size=64) do i
            i > 0.5 ? i : 0
        end
        @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))
    end
end
