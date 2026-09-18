struct Point
    x::Float32
    y::Float32
end
# Only for backend-agnostic initialisation with KernelAbstractions.zero
Base.zero(::Type{Point}) = Point(0.0f0, 0.0f0)

@testset "mapreduce_1d" begin
    Random.seed!(0)

    function minbox(s; prefer_threads)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            prefer_threads,
            init=(typemax(Float32), typemax(Float32)),
            neutral=(typemax(Float32), typemax(Float32)),
        )
    end

    function minbox_base(s)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        Base.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:num_elems])
        mgpu = minbox(v; prefer_threads)

        vh = Array(v)
        mcpu = minbox(vh; prefer_threads=true)
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)

        v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:n1, _ in 1:n2, _ in 1:n3])
        mgpu = minbox(v; prefer_threads)

        vh = Array(v)
        mcpu = minbox(vh; prefer_threads=true)
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # Ensuring that the init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32(1):Int32(100), num_elems))
        s = AK.mapreduce(abs, +, v; prefer_threads, init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(-100:-1, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.mapreduce(abs, +, v; prefer_threads, switch_below=switch_below, init=Int32(init))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh; init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.mapreduce(abs, +, v, BACKEND; prefer_threads, init=Int32(0))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh)
    end

    # Base-compatible alias: dims=: reduces all dimensions to a scalar.
    vh_colon = rand(Int32(-10):Int32(10), 3, 4, 5)
    @test AK.mapreduce(abs, +, array_from_host(vh_colon); prefer_threads, init=Int32(0), dims=:) ==
        mapreduce(abs, +, vh_colon; init=Int32(0), dims=:)

    vh_one = Int32[-7]
    @test AK.mapreduce(abs, +, array_from_host(vh_one); prefer_threads, init=Int32(10)) ==
        mapreduce(abs, +, vh_one; init=Int32(10))

    if !prefer_threads
        for len in (65, 257, 1025), items_per_thread in (1, 2, 4)
            vh = Int32.(mod.(1:len, 17) .- 8)
            v = array_from_host(vh)
            for (f, op, neutral) in ((x -> 3x - 1, +, Int32(0)),
                                     (abs, max, typemin(Int32)),
                                     (x -> -x, min, typemax(Int32)))
                @test AK.mapreduce(f, op, v; prefer_threads, init=neutral, neutral,
                                   block_size=64, items_per_thread) ==
                    Base.mapreduce(f, op, vh; init=neutral)
            end
        end

        @test_throws ArgumentError AK.mapreduce(identity, +, array_from_host(Int32[1, 2]);
                                                prefer_threads, init=Int32(0), items_per_thread=0)
    end

    vh_typechange = rand(Int32(-10):Int32(10), 4, 5)
    f_typechange = x -> Float32(x) / 2
    @test AK.mapreduce(f_typechange, +, array_from_host(vh_typechange); prefer_threads, init=0f0) ≈
        mapreduce(f_typechange, +, vh_typechange; init=0f0)
    @test Array(AK.mapreduce(f_typechange, +, array_from_host(vh_typechange); prefer_threads, init=0f0, dims=2)) ≈
        mapreduce(f_typechange, +, vh_typechange; init=0f0, dims=2)
    f_min_typechange = x -> Float32(10_000_000_000 + x)
    f_max_typechange = x -> Float32(-10_000_000_000 + x)
    @test AK.mapreduce(f_min_typechange, min, array_from_host(vh_typechange); prefer_threads, init=Inf32) ≈
        mapreduce(f_min_typechange, min, vh_typechange; init=Inf32)
    @test AK.mapreduce(f_max_typechange, max, array_from_host(vh_typechange); prefer_threads, init=-Inf32) ≈
        mapreduce(f_max_typechange, max, vh_typechange; init=-Inf32)

    # Multi-input mapreduce lowers through a broadcasted source.
    vh_a = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_b = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_c = rand(Int32(-10):Int32(10), 4, 5, 6)
    v_a = array_from_host(vh_a)
    v_b = array_from_host(vh_b)
    v_c = array_from_host(vh_c)
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b; prefer_threads, init=Int32(0)) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0))
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b, BACKEND; prefer_threads, init=Int32(0)) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0))
    @test AK.mapreduce((x, y, z) -> x + y * z, +, v_a, v_b, v_c, BACKEND; prefer_threads, init=Int32(0)) ==
        mapreduce((x, y, z) -> x + y * z, +, vh_a, vh_b, vh_c; init=Int32(0))
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b; prefer_threads, init=Int32(0), dims=:) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0), dims=:)
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_a, v_b; prefer_threads, init=Int32(0), dims=())) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0), dims=())
    @test AK.mapreduce((x, y) -> Float32(x - y) / 3, +, v_a, v_b; prefer_threads, init=0f0) ≈
        mapreduce((x, y) -> Float32(x - y) / 3, +, vh_a, vh_b; init=0f0)

    for (shape, dims) in (((0, 3), 1), ((2, 0), 2), ((0, 0), (1, 2)), ((0, 3), ()))
        h_empty1 = reshape(Int32[], shape...)
        h_empty2 = fill(Int32(2), shape...)
        @test Array(AK.mapreduce((x, y) -> x + y, +,
                                  array_from_host(h_empty1),
                                  array_from_host(h_empty2);
                                  prefer_threads, init=Int32(10), dims)) ==
            mapreduce((x, y) -> x + y, +, h_empty1, h_empty2; init=Int32(10), dims)
    end

    @test_throws DimensionMismatch AK.mapreduce(
        (x, y) -> x + y, +,
        array_from_host(rand(Int32, 2, 3)),
        array_from_host(rand(Int32, 1, 3));
        prefer_threads,
        init=Int32(0),
    )

    if prefer_threads
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(+, reshape(1:6, 2, 3), reshape(10:15, 2, 3)))
        @test AK.mapreduce(identity, +, bc; prefer_threads, init=0) ==
            mapreduce(identity, +, bc; init=0)
        @test Array(AK.mapreduce(identity, +, bc; prefer_threads, init=0, dims=2)) ==
            mapreduce(identity, +, bc; init=0, dims=2)
        @test Array(AK.mapreduce(identity, +, bc; prefer_threads, init=0, dims=())) ==
            mapreduce(identity, +, bc; init=0, dims=())
    end

    # Testing different settings, enforcing change of type between f and op
    f(s, temp) = AK.mapreduce(
        p -> (p.x, p.y),
        (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
        s;
        prefer_threads,
        init=(typemax(Float32), typemax(Float32)),
        neutral=(typemax(Float32), typemax(Float32)),
        block_size=64,
        temp=temp,
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:10_042])
    temp = similar(v, Tuple{Float32, Float32})
    f(v, temp)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.mapreduce(-, +, v; prefer_threads, init=10, bad=:kwarg)
    if !prefer_threads
        @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 256)); prefer_threads, init=Int32(0), block_size=192)
    end
end
