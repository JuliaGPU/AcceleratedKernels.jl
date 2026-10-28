@testset "reduce: whole arrays" begin
    Random.seed!(0)

    function redmin(s)
        # Reduction-based minimum finder
        AK.reduce(
            (x, y) -> x < y ? x : y,
            s;
            alg=REDUCE_ALG,
            init=typemax(eltype(s)),
            neutral=typemax(eltype(s)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        s = redmin(v)
        vh = Array(v)
        @test s == minimum(vh)
    end

    function redsum(s)
        # Reduction-based summation
        AK.reduce(
            (x, y) -> x + y,
            s;
            alg=REDUCE_ALG,
            init=zero(eltype(s)),
            neutral=zero(eltype(s)),
        )
    end

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), Int32)
        s = redsum(v)
        vh = Array(v)
        @test s == sum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), UInt32)
        s = redsum(v)
        vh = Array(v)
        @test s == sum(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        s = redsum(v)
        vh = Array(v)
        @test s ≈ sum(vh)
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)
        vh = rand(Float32, n1, n2, n3)
        v = array_from_host(vh)
        s = redsum(v)
        @test s ≈ sum(vh)
    end

    # The init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32(1):Int32(100), num_elems))
        s = AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.reduce(+, v; alg=reduce_alg(switch_below=switch_below), init=Int32(init))
        vh = Array(v)
        @test s == reduce(+, vh; init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.reduce(+, v; alg=REDUCE_ALG, backend=BACKEND, init=Int32(0))
        vh = Array(v)
        @test s == reduce(+, vh)
    end

    # Base-compatible alias: dims=: reduces all dimensions to a scalar.
    vh_colon = rand(Int32(1):Int32(10), 3, 4, 5)
    @test AK.reduce(+, array_from_host(vh_colon); alg=REDUCE_ALG, init=Int32(0), dims=:) ==
        reduce(+, vh_colon; init=Int32(0), dims=:)

    vh_one = Int32[7]
    @test AK.reduce(+, array_from_host(vh_one); alg=REDUCE_ALG, init=Int32(10)) ==
        reduce(+, vh_one; init=Int32(10))

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.reduce(+, array_from_host(rand(Int32, 10)); init=10, bad=:kwarg)
    if TEST_KERNELS
        @test_throws ArgumentError AK.reduce(+, array_from_host(rand(Int32, 256)); alg=reduce_alg(block_size=192), init=Int32(0))
    end

    # Testing different settings
    with_workspace(AK.reduce, (x, y) -> x + 1, array_from_host(rand(Int32, 10_000));
                   alg=reduce_alg(block_size=64, switch_below=50, max_tasks=10, min_elems=100), init=Int32(0), neutral=Int64(0))
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 10_000));
        alg=reduce_alg(max_tasks=16, min_elems=1000),
        init=Int32(0),
        neutral=Int64(0),
    )
end


@testset "mapreduce: whole arrays" begin
    Random.seed!(0)

    function minbox(s; alg=REDUCE_ALG)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            alg,
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
        mgpu = minbox(v; alg=REDUCE_ALG)

        vh = Array(v)
        mcpu = minbox(vh; alg=AK.Auto())
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
        mgpu = minbox(v; alg=REDUCE_ALG)

        vh = Array(v)
        mcpu = minbox(vh; alg=AK.Auto())
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # The init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32(1):Int32(100), num_elems))
        s = AK.mapreduce(abs, +, v; alg=REDUCE_ALG, init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(-100:-1, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.mapreduce(abs, +, v; alg=reduce_alg(switch_below=switch_below), init=Int32(init))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh; init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.mapreduce(abs, +, v; alg=REDUCE_ALG, backend=BACKEND, init=Int32(0))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh)
    end

    # Base-compatible alias: dims=: reduces all dimensions to a scalar.
    vh_colon = rand(Int32(-10):Int32(10), 3, 4, 5)
    @test AK.mapreduce(abs, +, array_from_host(vh_colon); alg=REDUCE_ALG, init=Int32(0), dims=:) ==
        mapreduce(abs, +, vh_colon; init=Int32(0), dims=:)

    vh_one = Int32[-7]
    @test AK.mapreduce(abs, +, array_from_host(vh_one); alg=REDUCE_ALG, init=Int32(10)) ==
        mapreduce(abs, +, vh_one; init=Int32(10))

    if TEST_KERNELS
        for len in (65, 257, 1025), items_per_thread in (1, 2, 4)
            vh = Int32.(mod.(1:len, 17) .- 8)
            v = array_from_host(vh)
            for (f, op, neutral) in ((x -> 3x - 1, +, Int32(0)),
                                     (abs, max, typemin(Int32)),
                                     (x -> -x, min, typemax(Int32)))
                @test AK.mapreduce(
                                   f,
                                   op,
                                   v;
                                   alg=reduce_alg(block_size=64, items_per_thread=items_per_thread),
                                   init=neutral,
                                   neutral,
) ==
                    Base.mapreduce(f, op, vh; init=neutral)
            end
        end

        @test_throws ArgumentError AK.mapreduce(
                                                identity,
                                                +,
                                                array_from_host(Int32[1, 2]);
                                                alg=reduce_alg(items_per_thread=0),
                                                init=Int32(0),
)
    end

    vh_typechange = rand(Int32(-10):Int32(10), 4, 5)
    f_typechange = x -> Float32(x) / 2
    @test AK.mapreduce(f_typechange, +, array_from_host(vh_typechange); alg=REDUCE_ALG, init=0f0) ≈
        mapreduce(f_typechange, +, vh_typechange; init=0f0)
    @test Array(AK.mapreduce(f_typechange, +, array_from_host(vh_typechange); alg=REDUCE_ALG, init=0f0, dims=2)) ≈
        mapreduce(f_typechange, +, vh_typechange; init=0f0, dims=2)
    f_min_typechange = x -> Float32(10_000_000_000 + x)
    f_max_typechange = x -> Float32(-10_000_000_000 + x)
    @test AK.mapreduce(f_min_typechange, min, array_from_host(vh_typechange); alg=REDUCE_ALG, init=Inf32) ≈
        mapreduce(f_min_typechange, min, vh_typechange; init=Inf32)
    @test AK.mapreduce(f_max_typechange, max, array_from_host(vh_typechange); alg=REDUCE_ALG, init=-Inf32) ≈
        mapreduce(f_max_typechange, max, vh_typechange; init=-Inf32)

    # Multi-input mapreduce lowers through a broadcasted source.
    vh_a = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_b = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_c = rand(Int32(-10):Int32(10), 4, 5, 6)
    v_a = array_from_host(vh_a)
    v_b = array_from_host(vh_b)
    v_c = array_from_host(vh_c)
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b; alg=REDUCE_ALG, init=Int32(0)) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0))
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b; alg=REDUCE_ALG, backend=BACKEND, init=Int32(0)) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0))
    @test AK.mapreduce((x, y, z) -> x + y * z, +, v_a, v_b, v_c; alg=REDUCE_ALG, backend=BACKEND, init=Int32(0)) ==
        mapreduce((x, y, z) -> x + y * z, +, vh_a, vh_b, vh_c; init=Int32(0))
    @test AK.mapreduce((x, y) -> x * y, +, v_a, v_b; alg=REDUCE_ALG, init=Int32(0), dims=:) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0), dims=:)
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_a, v_b; alg=REDUCE_ALG, init=Int32(0), dims=())) ==
        mapreduce((x, y) -> x * y, +, vh_a, vh_b; init=Int32(0), dims=())
    @test AK.mapreduce((x, y) -> Float32(x - y) / 3, +, v_a, v_b; alg=REDUCE_ALG, init=0f0) ≈
        mapreduce((x, y) -> Float32(x - y) / 3, +, vh_a, vh_b; init=0f0)
    for (max_tasks, min_elems) in ((4, 1), (7, 10), (16, 100))
        @test AK.mapreduce((x, y, z) -> x * (y - z), +, v_a, v_b, v_c;
                           alg=reduce_alg(; block_size=64, max_tasks, min_elems), init=Int32(3)) ==
            mapreduce((x, y, z) -> x * (y - z), +, vh_a, vh_b, vh_c; init=Int32(3))
    end

    for (shape, dims) in (((0, 3), 1), ((2, 0), 2), ((0, 0), (1, 2)), ((0, 3), ()))
        h_empty1 = reshape(Int32[], shape...)
        h_empty2 = fill(Int32(2), shape...)
        @test Array(AK.mapreduce(
                                  (x, y) -> x + y,
                                  +,
                                  array_from_host(h_empty1),
                                  array_from_host(h_empty2);
                                  alg=REDUCE_ALG,
                                  init=Int32(10),
                                  dims,
)) ==
            mapreduce((x, y) -> x + y, +, h_empty1, h_empty2; init=Int32(10), dims)
    end

    @test_throws DimensionMismatch AK.mapreduce(
        (x, y) -> x + y,
        +,
        array_from_host(rand(Int32, 2, 3)),
        array_from_host(rand(Int32, 1, 3));
        alg=REDUCE_ALG,
        init=Int32(0),
    )

    if !TEST_KERNELS
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(+, reshape(1:6, 2, 3), reshape(10:15, 2, 3)))
        @test AK.mapreduce(identity, +, bc; alg=REDUCE_ALG, init=0) ==
            mapreduce(identity, +, bc; init=0)
        @test Array(AK.mapreduce(identity, +, bc; alg=REDUCE_ALG, init=0, dims=2)) ==
            mapreduce(identity, +, bc; init=0, dims=2)
        @test Array(AK.mapreduce(identity, +, bc; alg=REDUCE_ALG, init=0, dims=())) ==
            mapreduce(identity, +, bc; init=0, dims=())
    end

    # Testing different settings, enforcing change of type between f and op
    f(s) = with_workspace(
        AK.mapreduce,
        p -> (p.x, p.y),
        (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
        s;
        alg=reduce_alg(block_size=64, switch_below=50, max_tasks=10, min_elems=100),
        init=(typemax(Float32), typemax(Float32)),
        neutral=(typemax(Float32), typemax(Float32)),
    )
    v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:10_042])
    f(v)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.mapreduce(-, +, v; alg=REDUCE_ALG, init=10, bad=:kwarg)
    if TEST_KERNELS
        @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 256)); alg=reduce_alg(block_size=192), init=Int32(0))
    end
end


@testset "sum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.sum(v; alg=REDUCE_ALG) == sum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.sum(v; alg=REDUCE_ALG) ≈ sum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.sum(v; alg=REDUCE_ALG) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v; alg=REDUCE_ALG, dims))
            rh = sum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.sum(v; alg=reduce_alg(block_size=64))
    @test AK.sum(v; alg=REDUCE_ALG, dims=:) == sum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.sum(v; alg=REDUCE_ALG, bad=:kwarg)

    # Element types without `zero(T)`, which only an empty sum needs
    if !TEST_KERNELS
        ms = [ones(2, 2), ones(2, 2)]
        @test AK.sum(ms; alg=REDUCE_ALG) == sum(ms)
        @test AK.prod(ms; alg=REDUCE_ALG, commutative=false) == prod(ms)
        @test AK.sum(Matrix{Float64}[]; init=zeros(2, 2), alg=REDUCE_ALG) == zeros(2, 2)
    end

    # The other settings are stress-tested in reduce
end


@testset "prod" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.prod(v; alg=REDUCE_ALG) == prod(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.prod(v; alg=REDUCE_ALG) ≈ prod(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.prod(v; alg=REDUCE_ALG) == prod(vh)

            # Along dimensions
            r = Array(AK.prod(v; alg=REDUCE_ALG, dims))
            rh = prod(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.prod(v; alg=reduce_alg(block_size=64))
    @test AK.prod(v; alg=REDUCE_ALG, dims=:) == prod(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.prod(v; alg=REDUCE_ALG, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "minimum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.minimum(v; alg=REDUCE_ALG) == minimum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.minimum(v; alg=REDUCE_ALG) == minimum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.minimum(v; alg=REDUCE_ALG) == minimum(vh)

            # Along dimensions
            r = Array(AK.minimum(v; alg=REDUCE_ALG, dims))
            rh = minimum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.minimum(v; alg=reduce_alg(block_size=64))
    @test AK.minimum(v; alg=REDUCE_ALG, dims=:) == minimum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.minimum(v; alg=REDUCE_ALG, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "maximum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.maximum(v; alg=REDUCE_ALG) == maximum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.maximum(v; alg=REDUCE_ALG) == maximum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.maximum(v; alg=REDUCE_ALG) == maximum(vh)

            # Along dimensions
            r = Array(AK.maximum(v; alg=REDUCE_ALG, dims))
            rh = maximum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.maximum(v; alg=reduce_alg(block_size=64))
    @test AK.maximum(v; alg=REDUCE_ALG, dims=:) == maximum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.maximum(v; alg=REDUCE_ALG, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "count" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.count(x->x>50, v; alg=REDUCE_ALG) == count(x->x>50, Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.count(x->x>0.5, v; alg=REDUCE_ALG) == count(x->x>0.5, Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.count(x->x>0.5, v; alg=REDUCE_ALG) == count(x->x>0.5, vh)

            # Along dimensions
            r = Array(AK.count(x->x>0.5, v; alg=REDUCE_ALG, dims))
            rh = count(x->x>0.5, vh; dims)

            @test r == rh
        end
    end

    # Counting booleans directly
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Bool, num_elems))
        @test AK.count(v; alg=REDUCE_ALG) == count(Array(v))
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.count(x->x>0, v; alg=reduce_alg(block_size=64))
    @test AK.count(x->x>0, v; alg=REDUCE_ALG, dims=:) == count(x->x>0, Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.count(v; alg=REDUCE_ALG, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "reduce: lazy sources" begin
    # Sources without a dense buffer, a lazy array and a fused one, reduced into one output and
    # with every launch shape along `dims`: few outputs of many elements each (in one block per
    # output, several, or one block for a few), contiguous outputs of strided elements (in
    # columns, or tiles), non-adjacent dimensions, and many outputs of few elements each
    Random.seed!(0)
    alg = REDUCE_ALG
    for (shape, dims) in (((64, 64, 64), (1, 2, 3)), ((64, 64, 64), (1, 2)),
                          ((300, 300, 3), (1, 2)), ((200, 5), 1), ((64, 64, 64), (2, 3)),
                          ((256, 256), 2), ((64, 64, 64), (1, 3)), ((64, 64, 64), 3))
        ha, hb = rand(Int32(-9):Int32(9), shape...), rand(Int32(-9):Int32(9), shape...)
        da, db = array_from_host(ha), array_from_host(hb)
        lazy, h = LazyProduct(da, db), ha .* hb
        @test Array(AK.reduce(+, lazy; dims, alg)) == sum(h; dims)
        @test Array(AK.reduce(+, lazy; dims, init=Int32(7), alg)) == reduce(+, h; dims, init=7)
        # An operator without a neutral element
        @test Array(AK.reduce((a, b) -> max(a, b), lazy; dims, alg)) == maximum(h; dims)
        @test Array(AK.mapreduce(*, +, da, db; dims, alg)) == sum(h; dims)
        @test Array(AK.mapreduce(*, +, lazy, lazy; dims, alg)) == sum(h .* h; dims)
        @test AK.mapreduce(*, +, lazy, lazy; alg) == sum(h .* h)
        # Folding into the destination's values, and overwriting them
        hR = rand(Int32(-9):Int32(9), size(sum(h; dims)))
        R = array_from_host(hR)
        AK.mapreducedim!(identity, +, R, lazy; alg)
        @test Array(R) == hR .+ sum(h; dims)
        AK.mapreducedim!(abs, max, R, lazy; overwrite=true, alg)
        @test Array(R) == maximum(abs, h; dims)
    end

    # Reductions into one or few outputs split each output's reduction across blocks, and keep
    # their partial results
    if TEST_KERNELS
        lazy = LazyProduct(array_from_host(ones(Int32, 300, 300, 3)),
                           array_from_host(ones(Int32, 300, 300, 3)))
        for R in (array_from_host(zeros(Int32, 1, 1, 1)), array_from_host(zeros(Int32, 1, 1, 3)))
            @test haskey(AK.workspace_size(AK.mapreducedim!, identity, +, R, lazy;
                                           alg=AK.BlockReduce()), :partials)
        end
    end
end


@testset "reduce: offset axes" begin
    # Arrays whose axes do not start at 1, as sources and destinations
    Random.seed!(0)
    algs = TEST_KERNELS ? [REDUCE_ALG] :
        [REDUCE_ALG, AK.CPUThreads.Partitioned(max_tasks=4, min_elems=1)]
    for alg in algs, commutative in (true, false)
        h = rand(Int32(-9):Int32(9), 10_000)
        v = offset_from_host(h, (-3,))
        @test AK.reduce(+, v; alg, commutative) == sum(h)
        @test AK.mapreduce(*, +, v, v; alg, commutative) == sum(h .* h)
        @test Array(AK.reduce(+, v; dims=1, alg, commutative)) == [sum(h)]
        single = offset_from_host(Int32[5], (-3,))
        @test AK.reduce(+, single; alg, commutative) == 5
        @test Array(AK.reduce(+, single; dims=1, alg, commutative)) == [5]

        hm = rand(Int32(-9):Int32(9), 30, 40)
        m = offset_from_host(hm, (-1, 3))
        @test AK.reduce(+, m; alg, commutative) == sum(hm)
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(abs, m))
        @test AK.mapreduce(identity, +, bc; alg, commutative) == sum(abs, hm)
        # A Broadcasted with explicit offset axes, larger than its argument's
        bc1 = Base.Broadcast.Broadcasted(identity, (array_from_host(Int32[2]),),
                                         (Base.IdentityUnitRange(0:999),))
        @test AK.mapreduce(identity, +, bc1; alg, commutative) == 2000
        @test Array(AK.mapreduce(identity, +, bc1; dims=1, alg, commutative)) == [2000]
        for dims in (1, 2, (1, 2))
            @test Array(AK.reduce(+, m; dims, alg, commutative)) == sum(hm; dims)
            @test Array(AK.mapreduce(*, +, m, m; dims, alg, commutative)) == sum(hm .* hm; dims)
            @test Array(AK.mapreduce(identity, +, bc; dims, alg, commutative)) ==
                  sum(abs, hm; dims)
            @test Array(with_workspace(AK.reduce, +, m; dims, alg, commutative)) == sum(hm; dims)

            # Into destinations with and without offset axes
            hR = rand(Int32(-9):Int32(9), size(sum(hm; dims)))
            R = array_from_host(hR)
            AK.mapreducedim!(identity, +, R, m; alg, commutative)
            @test Array(R) == hR .+ sum(hm; dims)
            R = offset_from_host(hR, (2, -5))
            AK.mapreducedim!(identity, +, R, m; alg, commutative)
            @test Array(R.parent) == hR .+ sum(hm; dims)
            with_workspace(AK.mapreducedim!, identity, max, R, m; overwrite=true, alg,
                           commutative)
            @test Array(R.parent) == maximum(hm; dims)
        end

        # Empty slices set the outputs to `init`
        R = offset_from_host(zeros(Int32, 3, 1), (2, 2))
        AK.mapreducedim!(identity, +, R, offset_from_host(zeros(Int32, 3, 0), (2, 2));
                         init=Int32(7), alg, commutative)
        @test Array(R.parent) == fill(7, 3, 1)
    end
end


@testset "reduce: contract" begin
    Random.seed!(0)
    alg = REDUCE_ALG

    # Omitted `init`: an empty reduction is an error, except for `sum`, `prod` and `count`, which
    # give zero or one of the accumulator type
    @test AK.sum(array_from_host(Int32[]); alg) === 0
    @test AK.sum(array_from_host(Float32[]); alg) === 0.0f0
    @test AK.sum(array_from_host(Int32[]); acctype=Int32, alg) === Int32(0)
    @test AK.prod(array_from_host(Int32[]); alg) === 1
    @test AK.count(array_from_host(Bool[]); alg) === 0
    @test_throws ArgumentError AK.minimum(array_from_host(Int32[]); alg)
    @test_throws ArgumentError AK.reduce(max, array_from_host(Int32[]); alg)
    @test_throws ArgumentError AK.reduce(+, array_from_host(Int32[]); alg)
    @test_throws ArgumentError AK.reduce((a, b) -> a + b, array_from_host(Int32[]); alg)
    # An explicit `init` is returned as it is for an empty reduction, and applied once otherwise,
    # also when it is not a neutral element
    @test AK.sum(array_from_host(Float32[]); init=1, alg) === 1
    @test AK.maximum(array_from_host(Int32[]); init=Int32(-1), alg) === Int32(-1)
    @test AK.sum(array_from_host(ones(Int32, 1000)); init=Int32(10), alg) === 1010
    @test AK.reduce(*, array_from_host(fill(Int32(2), 20)); init=Int32(3), alg) === Int32(3 << 20)
    # `init=nothing` is an initial value, not an omitted one; `+` cannot fold it, which only
    # matters when there is something to fold
    @test_throws Exception AK.reduce(+, array_from_host(Int32[1, 2]); init=nothing, alg)
    @test AK.reduce(+, array_from_host(Int32[]); init=nothing, alg) === nothing
    @test AK.mapreduce(x -> error("never called"), +, array_from_host(Int32[]); init=7, alg) === 7

    # The accumulator type is the fold type of `op`, from `init`'s type and the mapped elements
    h8 = Int8[100, 100, 27]
    v8 = array_from_host(h8)
    @test AK.sum(v8; alg) === 227
    @test AK.reduce(+, v8; alg) === Int8(-29)
    @test AK.reduce(+, v8; init=0, alg) === 227
    @test AK.sum(array_from_host([1, 2]); init=Int8(0), alg) === 3
    @test AK.reduce(+, array_from_host(Float32[1, 2]); init=0, alg) === 3.0f0
    # `init` is not a partial result: only the type of `op(init, x)` counts
    @test AK.reduce((a, b) -> something(a, 0) + something(b, 0), array_from_host(Int32[1, 2]);
                    init=nothing, alg) === 3
    @test AK.reduce((a, b) -> Base.add_sum(a, b), v8; alg) === 227
    @test AK.maximum(v8; alg) === Int8(100)
    # It joins the type of the one-element partial results: here the element type, not the
    # narrower one the fold produces
    @test AK.reduce(Returns(Int8(0)), array_from_host(Int32[5, 6, 7]); alg) === Int32(0)
    if !TEST_KERNELS
        @test AK.reduce(coalesce, [missing, 1, 2]; init=0, alg) === 0
    end
    # Partial results of small integers are `Int`s, also for mixed signedness (where Julia 1.10's
    # `add_sum` of the `init` and an element would stay a `UInt8`)
    @test AK.sum(array_from_host(Int8[-1, -2]); init=UInt8(0), alg) === -3
    @test AK.count(array_from_host([true, true]); init=UInt8(0), alg) === 2
    # A single element has the accumulator type too; `f` may index device arrays
    @test AK.reduce((a, b) -> a + b, array_from_host([true]); alg) === 1
    @test AK.sum(array_from_host([true]); alg) === 1
    w = array_from_host(Int32[11, 22])
    @test AK.mapreduce(let w = w; x -> w[x]; end, +, array_from_host(Int32[2]); alg) === Int32(22)
    # Seeds are exact identities, and `sum`'s empty rule is no `init`: a sum of negative zeros is
    # a negative zero
    for n in (1, 2, 1000)
        @test AK.sum(array_from_host(fill(-0.0f0, n)); alg) === -0.0f0
        @test AK.sum(array_from_host(fill(complex(-0.0f0, -0.0f0), n)); alg) ===
              complex(-0.0f0, -0.0f0)
        @test Base.all(signbit, Array(AK.sum(array_from_host(fill(-0.0f0, 2, n)); dims=2, alg)))
    end

    # `acctype` sets the accumulator type, and the result's
    @test AK.reduce(+, array_from_host(Int8[100, 100, 56]); acctype=Int8, alg) === Int8(0)
    @test AK.sum(array_from_host(Int8[100, 100, 56]); acctype=Int16, alg) === Int16(256)
    @test AK.reduce(max, v8; init=Int8(0), acctype=Int32, alg) === Int32(100)
    xf = Float32[1.0f8, 1, -1.0f8]
    if Float64 in valid_backend_eltypes(BACKEND, (Float64,))
        @test AK.sum(array_from_host(xf); acctype=Float64, alg) === 1.0
        R = array_from_host(zeros(Float32, 1, 2))
        AK.mapreducedim!(identity, +, R, array_from_host([xf xf]); overwrite=true,
                         acctype=Float64, alg)
        @test Array(R) == [1 1]
    end
    m8 = rand(Int8(-9):Int8(9), 40, 30)
    dm8 = array_from_host(m8)
    for dims in (1, 2)
        r = AK.sum(dm8; dims, acctype=Int16, alg)
        @test eltype(r) === Int16 && Array(r) == sum(m8; dims)
    end
    # `init` does not change it; only an empty reduction returns `init` as it is
    @test AK.sum(array_from_host(Int32[1, 2]); init=0, acctype=Int32, alg) === Int32(3)
    @test AK.count(array_from_host([true, true]); acctype=Int32, alg) === Int32(2)
    @test AK.sum(array_from_host(Int32[]); init=0, acctype=Int32, alg) === 0
    if !TEST_KERNELS
        @test_throws InexactError AK.reduce(+, Int8[1, 2]; init=0.5, acctype=Int16, alg)
    end
    # It must be able to hold the partial results
    @test_throws ArgumentError AK.sum(v8; acctype=String, alg)
    @test_throws ArgumentError AK.reduce((a, b) -> "", v8; acctype=Int, alg)
    @test_throws ArgumentError AK.sum(v8; acctype=1, alg)
    # Whether their values fit is the caller's obligation
    if !TEST_KERNELS
        @test_throws InexactError AK.sum([100, 100]; acctype=Int8, alg)
    end

    # Reductions along `dims` return the accumulator type, also with `init`
    for dims in (1, 2, (1, 2), 3)
        r = AK.sum(dm8; dims, alg)
        @test eltype(r) === Int && Array(r) == sum(m8; dims)
        r = AK.sum(dm8; dims, init=Int16(1), alg)
        @test eltype(r) === Int && Array(r) == sum(m8; dims, init=Int16(1))
        r = AK.maximum(dm8; dims, alg)
        @test eltype(r) === Int8 && Array(r) == maximum(m8; dims)
        r = AK.count(x -> x > 0, dm8; dims, alg)
        @test eltype(r) === Int && Array(r) == count(x -> x > 0, m8; dims)
    end

    # Empty reduced dimensions: `init`, else an error, except for `sum`, `prod` and `count`
    e8 = array_from_host(zeros(Int8, 0, 3))
    for (r, value) in ((AK.sum(e8; dims=1, alg), 0), (AK.prod(e8; dims=1, alg), 1),
                       (AK.count(x -> x > 0, e8; dims=1, alg), 0))
        @test eltype(r) === Int && Array(r) == fill(value, 1, 3)
    end
    @test eltype(AK.sum(e8; dims=1, acctype=Int8, alg)) === Int8
    @test Array(AK.minimum(e8; dims=1, init=Int8(7), alg)) == fill(Int8(7), 1, 3)
    @test_throws ArgumentError AK.minimum(e8; dims=1, alg)
    @test_throws ArgumentError AK.mapreduce(x -> x + 1, +, array_from_host(zeros(Int32, 0, 2));
                                            dims=1, alg)
    @test size(AK.maximum(array_from_host(zeros(Int8, 3, 0)); dims=1, alg)) == (1, 0)
    # An error only where an output has an empty slice
    @test size(AK.minimum(array_from_host(zeros(Int32, 0, 0)); dims=1, alg)) == (1, 0)

    # An operator that always throws for the element types: an error only where something is
    # combined
    throws(a, b) = throw(ArgumentError("never called"))
    @test AK.reduce(throws, array_from_host(Int32[5]); alg) === Int32(5)
    x5 = array_from_host(Int32[5])
    @test AK.mapreduce(+, throws, x5, x5; alg) === Int32(10)
    @test Array(AK.reduce(throws, array_from_host(Int32[5 6]); dims=1, alg)) == [5 6]
    @test_throws ArgumentError AK.reduce(throws, array_from_host(Int32[5, 6]); alg)
    @test_throws ArgumentError AK.reduce(throws, array_from_host(Int32[5, 6]); dims=1, alg)
    # On the host, an abstract accumulator type with the caller's neutral element, and partial
    # results of a type that is not a bits type without one
    if !TEST_KERNELS
        @test AK.reduce(+, Real[1, 2.5, 3]; neutral=0, alg=AK.CPUThreads.Partitioned(max_tasks=2, min_elems=1)) == 6.5
        @test AK.reduce((a, b) -> a * b, fill("x", 8);
                        alg=AK.CPUThreads.Partitioned(max_tasks=4, min_elems=2)) == "x"^8
    end

    # Operators without a known neutral element: partial results start from their first element
    x = rand(Int32(-100):Int32(100), 10_001)
    dx = array_from_host(x)
    @test AK.reduce((a, b) -> a + b, dx; alg) == sum(x)
    @test AK.reduce((a, b) -> min(a, b), dx; alg) == minimum(x)
    @test AK.reduce((a, b) -> a + b, dx; init=Int32(10), alg) == sum(x) + 10
    @test AK.reduce((a, b) -> a + b, array_from_host(Int32[7]); alg) === Int32(7)
    m = rand(Int32(-100):Int32(100), 37, 300)
    for dims in (1, 2, (1, 2), 3)
        @test Array(AK.reduce((a, b) -> a + b, array_from_host(m); dims, alg)) == sum(m; dims)
        @test Array(AK.mapreduce(abs, (a, b) -> max(a, b), array_from_host(m); dims, alg)) ==
              maximum(abs, m; dims)
    end

    # Tuple and named-tuple accumulators, as GPUArrays uses for `findmin` and Missing-aware `==`
    xs = rand(Float32, 5000)
    ix = collect(Int32(1):Int32(5000))
    findmin_op(a, b) = (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])) ? a : b
    @test AK.mapreduce(tuple, findmin_op, array_from_host(xs), array_from_host(ix); alg) ==
          (minimum(xs), Int32(argmin(xs)))
    ms = rand(Float32, 20, 300)
    im = reshape(collect(Int32(1):Int32(length(ms))), size(ms))
    r = Array(AK.mapreduce(tuple, findmin_op, array_from_host(ms), array_from_host(im); dims=2, alg))
    @test r == reshape([(minimum(ms[i, :]), im[i, argmin(ms[i, :])]) for i in 1:20], 20, 1)
    eq_op(a, b) = (is_missing = a.is_missing | b.is_missing, is_equal = a.is_equal & b.is_equal)
    @test AK.mapreduce(x -> (is_missing = false, is_equal = x > -101), eq_op, dx; alg) ==
          (is_missing = false, is_equal = true)

    # `mapreducedim!`: fold into the destination, overwrite it, or apply `init` once
    A = array_from_host(Int32[1 3; 2 4])
    R = array_from_host(Int32[10 20])
    @test AK.mapreducedim!(identity, +, R, A; alg) === R
    @test Array(R) == [13 27]
    R = array_from_host(Int32[10 20])
    @test Array(AK.mapreducedim!(identity, +, R, A; overwrite=true, alg)) == [3 7]
    R = array_from_host(Int32[10 20])
    @test Array(AK.mapreducedim!(identity, +, R, A; init=Int32(100), alg)) == [103 107]
    R = array_from_host(Int32[10 20])
    @test Array(AK.mapreducedim!(identity, (a, b) -> a + b, R, A; alg)) == [13 27]
    R = array_from_host(Int32[10 20])
    @test Array(AK.mapreducedim!(x -> 2x, +, R, A; alg)) == [16 34]
    # A destination may leave off trailing singleton dimensions, and have extra ones
    @test Array(AK.mapreducedim!(identity, +, array_from_host(zeros(Int32, 2)), A;
                                 overwrite=true, alg)) == [4, 6]
    @test Array(AK.mapreducedim!(identity, +, array_from_host(zeros(Int32, 1, 2, 1)), A;
                                 overwrite=true, alg)) == reshape([3, 7], 1, 2, 1)
    # No reduced dimension: every output reduces one element
    R = array_from_host(Int32[1 1; 1 1])
    @test Array(AK.mapreducedim!(identity, +, R, A; alg)) == [2 4; 3 5]
    # Empty slices: `init`, else not written, when folding and overwriting alike
    E = array_from_host(zeros(Int32, 0, 2))
    R = array_from_host(Int32[10 20])
    @test Array(AK.mapreducedim!(identity, +, R, E; alg)) == [10 20]
    @test Array(AK.mapreducedim!(identity, +, R, E; overwrite=true, alg)) == [10 20]
    @test Array(AK.mapreducedim!(identity, min, R, E; overwrite=true, alg)) == [10 20]
    @test Array(AK.mapreducedim!(identity, min, R, E; init=Int32(5), alg)) == [5 5]
    # The accumulator type starts from the destination's element type
    R = array_from_host(zeros(Int16, 1, 3))
    AK.mapreducedim!(identity, +, R, array_from_host(fill(Int8(100), 4, 3)); overwrite=true, alg)
    @test Array(R) == fill(Int16(400), 1, 3)
    # A Broadcasted source
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(*, A, Int32(2)))
    R = array_from_host(zeros(Int32, 1, 2))
    @test Array(AK.mapreducedim!(identity, +, R, bc; overwrite=true, alg)) == [6 14]
    # Shapes and aliasing
    @test_throws DimensionMismatch AK.mapreducedim!(identity, +, array_from_host(zeros(Int32, 3, 1)), A; alg)
    @test_throws DimensionMismatch AK.mapreducedim!(identity, +, array_from_host(zeros(Int32, 1, 1, 2)), A; alg)
    @test_throws ArgumentError AK.mapreducedim!(identity, +, view(A, 1:1, :), A; alg)
    bc = Base.Broadcast.preprocess(nothing, Base.Broadcast.instantiate(
        Base.Broadcast.broadcasted(identity, A)))
    @test_throws ArgumentError AK.mapreducedim!(identity, +, view(A, 1:1, :), bc;
                                                overwrite=true, alg)
    # A preprocessed Broadcasted source (with `Extruded` arguments) keeps its element type
    R = array_from_host(zeros(Int32, 1, 2))
    @test Array(AK.mapreducedim!(identity, +, R, bc; overwrite=true, alg)) == [3 7]

    # Kernels need a bits-type accumulator
    if TEST_KERNELS
        @test_throws ArgumentError AK.mapreduce(x -> x > 0 ? x : missing, +, dx; alg)
    end

    # A source whose wrappers `@Const` rebuilds on the device: a reshaped view
    hr = rand(Int32(0):Int32(9), 50, 40)
    dr = array_from_host(hr)
    vr, hv = vec(view(dr, 1:40, 1:30)), vec(view(hr, 1:40, 1:30))
    @test AK.reduce(+, vr; alg) == sum(hv)
    @test AK.count(!iszero, vr; alg) == count(!iszero, hv)
    rr, hrr = reshape(view(dr, 1:40, 1:30), 30, 40), reshape(view(hr, 1:40, 1:30), 30, 40)
    for dims in (1, 2)
        @test Array(AK.reduce(+, rr; dims, alg)) == sum(hrr; dims)
    end

    # A source of a bits-union element type, with every kernel shape (where the backend's arrays
    # can hold one)
    code(x) = x === missing ? 0x01 : x ? 0x02 : 0x00
    TEST_BITS_UNIONS && for (shape, dims) in (((1000, 300), 1), ((1000, 300), 2), ((256, 256), 2),
                          ((20_000, 2), (1, 2)), ((5, 3), 3))
        hm = rand([true, false, missing], shape...)
        dm = array_from_host(hm)
        @test AK.mapreduce(code, max, dm; init=0x00, alg) === mapreduce(code, max, hm)
        @test Array(AK.mapreduce(code, max, dm; dims, init=0x00, alg)) ==
              mapreduce(code, max, hm; dims)
    end
end


@testset "reduce: partial results without a neutral element" begin
    # Such partial results are `Optional`s, which kernels load whole and the host copies, so a
    # partial must hold no bytes that its stores leave unwritten (Compute Sanitizer's initcheck
    # reports reading them). An `Optional` adds no padding to its value's type.
    for T in (Bool, Int8, Int32, Float32, Float64, ComplexF32, NTuple{3, Int32},
              Tuple{Float64, Float64})
        @test !Base.datatype_haspadding(typeof(AK.reduce_seed((a, b) -> a, T, nothing)))
    end
    # Every partial result a reduction stores is written in all its bytes or in none (a slot that
    # this length leaves unused), whatever its buffer held before
    if TEST_KERNELS
        alg = REDUCE_ALG
        max2(a, b) = max(a, b)
        pairmax(p, q) = (max(p[1], q[1]), max(p[2], q[2]))
        h = rand(Float32, 100_000)
        for (f, op, src, dims) in ((identity, max2, h, :), (identity, max2, h[1:100], :),
                                   (x -> (x, -x), pairmax, h, :),
                                   (identity, max2, reshape(h, :, 2), 1))
            v = array_from_host(src)
            ws = AK.workspace(AK.mapreduce, f, op, v; dims, alg)
            p = ws.buffers.partials
            b0, b1 = map((0x00, 0xff)) do poison
                fill!(reinterpret(UInt8, vec(p)), poison)
                AK.mapreduce(f, op, v; dims, alg, workspace=ws)
                reshape(Array(reinterpret(UInt8, vec(p))), sizeof(eltype(p)), :)
            end
            @test all(j -> allequal(b0[:, j] .== b1[:, j]), axes(b0, 2))
            @test any(j -> b0[:, j] == b1[:, j], axes(b0, 2))
        end
    end
end
