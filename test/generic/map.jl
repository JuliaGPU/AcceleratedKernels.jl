@testset "map" begin
    Random.seed!(0)

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
    # Tests different things with GPU and CPU backends as well as irrelevant parameters being ignored
    y = AK.map(x; prefer_threads, block_size=64, max_tasks=2, min_elems=100) do i
        i > 0.5 ? i : 0
    end
    @test Array(y) == map(i -> i > 0.5 ? i : 0, Array(x))

    if prefer_threads # CPU only
        x = rand(Float32, 1000)
        y = AK.map(x; prefer_threads, max_tasks=4, min_elems=500) do i
            i > 0.5 ? i : 0
        end
        @test y == map(i -> i > 0.5 ? i : 0, x)
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.map(x -> x^2, x; prefer_threads, bad=:kwarg)
end


@testset "map multi-arg" begin
    Random.seed!(0)

    a = array_from_host(rand(Float32, 1000))
    b = array_from_host(rand(Float32, 1000))
    c = array_from_host(rand(Float32, 1000))
    ah, bh, ch = Array(a), Array(b), Array(c)

    # two and three source arrays, out-of-place
    @test Array(AK.map((x, y) -> x + y, a, b; prefer_threads)) == map(+, ah, bh)
    @test Array(AK.map((x, y, z) -> x * y + z, a, b, c; prefer_threads)) ==
          map((x, y, z) -> x * y + z, ah, bh, ch)

    # eltype-changing multi-arg map
    @test Array(AK.map((x, y) -> x > y, a, b; prefer_threads)) == map((x, y) -> x > y, ah, bh)

    # in-place, multiple sources
    d = array_from_host(zeros(Float32, 1000))
    AK.map!((x, y) -> x - y, d, a, b; prefer_threads)
    @test Array(d) == map(-, ah, bh)

    # explicit trailing backend argument
    backend = get_backend(a)
    @test Array(AK.map(+, a, b, backend; prefer_threads)) == map(+, ah, bh)
    e = array_from_host(zeros(Float32, 1000))
    AK.map!(+, e, a, b, backend; prefer_threads)
    @test Array(e) == map(+, ah, bh)

    # mismatched axes are rejected
    @test_throws DimensionMismatch AK.map(+, a, array_from_host(rand(Float32, 999)); prefer_threads)
end
