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

    # Arrays of the same length but different shapes are matched in column-major order
    x = array_from_host(reshape(1:12, 3, 4))
    y = array_from_host(zeros(Int, 12))
    AK.map!(i -> 2i, y, x)
    @test Array(y) == 2:2:24
    y = array_from_host(zeros(Int, 4, 3))
    AK.map!(i -> 2i, y, view(x, :, :))
    @test vec(Array(y)) == 2:2:24
    # ... also through views that do not support linear indexing
    x = array_from_host(reshape(1:20, 4, 5))
    v = view(x, 1:3, 2:5)
    y = array_from_host(zeros(Int, 12))
    AK.map!(i -> 2i, y, v)
    @test Array(y) == 2 .* vec(Array(x)[1:3, 2:5])
    y = array_from_host(zeros(Int, 4, 5))
    AK.map!(i -> 2i, view(y, 2:4, 1:4), array_from_host(1:12))
    @test Array(y)[2:4, 1:4] == reshape(2:2:24, 3, 4)
    @test Array(y)[1, :] == zeros(Int, 5)

    # Sources whose indices do not start at 1
    y = array_from_host(zeros(Int, 3))
    AK.map!(+, y, Base.IdentityUnitRange(0:2), Base.IdentityUnitRange(0:2); backend=BACKEND)
    @test Array(y) == [0, 2, 4]

    # Empty arrays
    x = array_from_host(Int[])
    @test isempty(AK.map(i -> i + 1, x))
    @test AK.map!(+, x, x, x) === x

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


@testset "map: several sources" begin
    Random.seed!(0)

    a = array_from_host(rand(Float32, 1000))
    b = array_from_host(rand(Float32, 1000))
    c = array_from_host(rand(Float32, 1000))
    ah, bh, ch = Array(a), Array(b), Array(c)

    @test Array(AK.map(+, a, b)) == ah .+ bh
    @test Array(AK.map((x, y, z) -> x * y + z, a, b, c)) ≈ ah .* bh .+ ch
    @test Array(AK.map(+, a, b; block_size=64, max_tasks=2, min_elems=100)) == ah .+ bh

    d = array_from_host(zeros(Float32, 1000))
    @test AK.map!(-, d, a, b) === d
    @test Array(d) == ah .- bh

    # The result has the shape and eltype of the first source
    x = array_from_host(reshape(1:12, 3, 4))
    y = array_from_host(Float32.(1:12))
    r = AK.map(+, x, y)
    @test r isa AbstractMatrix{Int}
    @test Array(r) == reshape(2:2:24, 3, 4)
    r = AK.map(+, y, x)
    @test r isa AbstractVector{Float32}
    @test Array(r) == 2:2:24

    # Sources of different shapes and index styles
    x = array_from_host(reshape(1:20, 4, 5))
    v = view(x, 1:3, 2:5)
    w = array_from_host(1:12)
    d = array_from_host(zeros(Int, 2, 6))
    AK.map!((p, q, s) -> p + 10q + 100s, d, w, v, v)
    @test vec(Array(d)) == (1:12) .+ 110 .* vec(Array(x)[1:3, 2:5])

    # Backend-free sources run on the backend of the others, or the one given
    r = AK.map((i, x) -> x > 0.5f0 ? i : 0, 1:1000, a)
    @test get_backend(r) == get_backend(a)
    @test Array(r) == ifelse.(ah .> 0.5f0, 1:1000, 0)
    r = AK.map(+, 1:5, 1:5; backend=BACKEND)
    @test get_backend(r) == BACKEND
    @test Array(r) == 2:2:10

    # Mismatched lengths are rejected
    @test_throws ArgumentError AK.map(+, a, array_from_host(rand(Float32, 999)))
    @test_throws ArgumentError AK.map!(+, d, w, array_from_host(1:11))
end
