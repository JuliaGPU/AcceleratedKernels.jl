@testset "map" begin
    Random.seed!(0)

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
    # Tests different things with GPU and CPU backends as well as irrelevant parameters being ignored
    y = AK.map(x; block_size=64, max_tasks=2, min_elems=100) do i
        i > 0.5 ? i : 0
    end
    @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))

    if AK._runs_threads(BACKEND) # host arrays
        x = rand(Float32, 1000)
        y = AK.map(x; max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)
    end

    # map! returns its destination, which must have as many elements as the source
    x = array_from_host(1:1000)
    y = array_from_host(zeros(Int, 1000))
    @test AK.map!(i -> i + 1, y, x) === y
    @test_throws ArgumentError AK.map!(identity, array_from_host(zeros(Int, 3)), x)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.map(x -> x^2, x; bad=:kwarg)
    @test_throws MethodError AK.map(x -> x^2, x; prefer_threads=true)
end

@testset "map: backend-free inputs" begin
    # A range has no backend: the result is allocated on the one given
    r = AK.map(x -> 2x, 1:5; backend=BACKEND)
    @test get_backend(r) == BACKEND
    @test Array(r) == 2:2:10
end
