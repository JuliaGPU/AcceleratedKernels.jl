struct Point
    x::Float32
    y::Float32
end
# Only for backend-agnostic initialisation with KernelAbstractions.zero
Base.zero(::Type{Point}) = Point(0.0f0, 0.0f0)

@testset "reduce_1d" begin
    Random.seed!(0)

    function redmin(s)
        # Reduction-based minimum finder
        AK.reduce(
            (x, y) -> x < y ? x : y,
            s;
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

    # Ensuring that the init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32(1):Int32(100), num_elems))
        s = AK.reduce(+, v; init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.reduce(+, v; switch_below=switch_below, init=Int32(init))
        vh = Array(v)
        @test s == reduce(+, vh, init=init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.reduce(+, v, BACKEND; init=Int32(0))
        vh = Array(v)
        @test s == reduce(+, vh)
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 10_000)),
        init=Int32(0),
        neutral=Int64(0),
        block_size=64,
        temp=array_from_host(zeros(Int32, 10_000)),
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.reduce(
        (x, y) -> x + 1,
        rand(Int32, 10_000),
        init=Int32(0),
        neutral=Int64(0),
        max_tasks=16,
        min_elems=1000,
    )
end


@testset "reduce_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.reduce(+, s; init=Int32(10), dims=dims)
                    dh = Array(d)
                    @test dh == sum(sh, init=Int32(10), dims=dims)
                    @test eltype(dh) == eltype(sum(sh, init=Int32(10), dims=dims))
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
            s = AK.reduce(+, v; init=Int32(0), dims=dims)
            sh = Array(s)
            @test sh == sum(vh, dims=dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; init=UInt32(0), dims=dims)
            sh = Array(s)
            @test sh == sum(vh, dims=dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; init=Float32(0), dims=dims)
            sh = Array(s)
            @test sh ≈ sum(vh, dims=dims)
        end
    end

    # Ensuring that the init value is respected
    for _ in 1:100
        for dims in 1:4
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.reduce(+, v; init=Int32(init), dims=dims)
            sh = Array(s)
            @test sh == reduce(+, vh, dims=dims, init=init)
        end
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        neutral=Int32(0),
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 1, 5)),
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        neutral=Int32(0),
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 4, 1)),
        switch_below=50,
        max_tasks=16,
        min_elems=1000,
    )
end


@testset "mapreduce_1d" begin
    Random.seed!(0)

    function minbox(s)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
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
        mgpu = minbox(v)

        vh = Array(v)
        mcpu = minbox(vh)
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
        mgpu = minbox(v)

        vh = Array(v)
        mcpu = minbox(vh)
        mbase = minbox_base(vh)

        @test typeof(mgpu) === typeof(mcpu) === typeof(mbase)
        @test mgpu[1] ≈ mcpu[1] ≈ mbase[1]
        @test mgpu[2] ≈ mcpu[2] ≈ mbase[2]
    end

    # Ensuring that the init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32(1):Int32(100), num_elems))
        s = AK.mapreduce(abs, +, v; init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(-100:-1, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.mapreduce(abs, +, v; switch_below=switch_below, init=Int32(init))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh, init=init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.mapreduce(abs, +, v, BACKEND; init=Int32(0))
        vh = Array(v)
        @test s == mapreduce(abs, +, vh)
    end

    # Testing different settings, enforcing change of type between f and op
    f(s, temp) = AK.mapreduce(
        p -> (p.x, p.y),
        (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
        s,
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
end


@testset "mapreduce_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(-100):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.mapreduce(-, +, s; init=Int32(-10), dims=dims)
                    dh = Array(d)
                    @test dh == mapreduce(-, +, sh, init=Int32(-10), dims=dims)
                    @test eltype(dh) == eltype(mapreduce(-, +, sh, init=Int32(-10), dims=dims))
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
            s = AK.mapreduce(-, +, v; init=Int32(0), dims=dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh, init=Int32(0), dims=dims)
        end
    end

    function minbox(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
            neutral=(typemax(Float32), typemax(Float32)),
            dims=dims,
        )
    end

    function minbox_base(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        Base.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
            dims=dims,
        )
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            v = array_from_host([Point(rand(Float32), rand(Float32)) for _ in 1:n1, _ in 1:n2, _ in 1:n3])
            mgpu = minbox(v, dims)

            vh = Array(v)
            mcpu = minbox(vh, dims)
            mbase = minbox_base(vh, dims)

            @test eltype(mgpu) === eltype(mcpu) === eltype(mbase)
            @test all([
                (mgpu_red[1] ≈ mcpu[i][1] ≈ mbase[i][1]) && (mgpu_red[2] ≈ mcpu[i][2] ≈ mbase[i][2])
                for (i, mgpu_red) in enumerate(Array(mgpu))
            ])
        end
    end

    # Ensuring that the init value is respected
    for _ in 1:100
        for dims in 1:4
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(-100):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.mapreduce(-, +, v; init=Int32(init), dims=dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh, dims=dims, init=init)
        end
    end

    # Testing different settings
    AK.mapreduce(
        -,
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        neutral=Int32(0),
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 1, 5)),
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.mapreduce(
        -,
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5)),
        init=Int32(0),
        neutral=Int32(0),
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Int32, 3, 4, 1)),
        switch_below=50,
        max_tasks=16,
        min_elems=1000,
    )
end
@testset "sum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.sum(v) == sum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.sum(v) ≈ sum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.sum(v) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v, dims=dims))
            rh = sum(vh, dims=dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.sum(v, block_size=64)

    # The other settings are stress-tested in reduce
end


@testset "prod" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.prod(v) == prod(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.prod(v) ≈ prod(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.sum(v) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v, dims=dims))
            rh = sum(vh, dims=dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.prod(v, block_size=64)

    # The other settings are stress-tested in reduce
end


@testset "minimum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.minimum(v) == minimum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.minimum(v) == minimum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.minimum(v) == minimum(vh)

            # Along dimensions
            r = Array(AK.minimum(v, dims=dims))
            rh = minimum(vh, dims=dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.minimum(v, block_size=64)

    # The other settings are stress-tested in reduce
end


@testset "maximum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.maximum(v) == maximum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.maximum(v) == maximum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.maximum(v) == maximum(vh)

            # Along dimensions
            r = Array(AK.maximum(v, dims=dims))
            rh = maximum(vh, dims=dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.maximum(v, block_size=64)

    # The other settings are stress-tested in reduce
end


@testset "count" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.count(x->x>50, v) == count(x->x>50, Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.count(x->x>0.5, v) == count(x->x>0.5, Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.count(x->x>0.5, v) == count(x->x>0.5, vh)

            # Along dimensions
            r = Array(AK.count(x->x>0.5, v, dims=dims))
            rh = count(x->x>0.5, vh, dims=dims)

            @test r == rh
        end
    end

    # Counting booleans directly
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Bool, num_elems))
        @test AK.count(v) == count(Array(v))
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.count(x->x>0, v, block_size=64)

    # The other settings are stress-tested in reduce
end
