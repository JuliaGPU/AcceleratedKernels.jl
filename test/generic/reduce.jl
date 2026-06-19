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
            prefer_threads,
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
            prefer_threads,
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
        s = AK.reduce(+, v; prefer_threads, init=Int32(10))
        vh = Array(v)
        @test s == sum(vh) + 10
    end

    # Testing with switch_below - i.e. finishing on the CPU
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(1:100, num_elems), Int32)
        switch_below = rand(1:100)
        init = rand(1:100)
        s = AK.reduce(+, v; prefer_threads, switch_below=switch_below, init=Int32(init))
        vh = Array(v)
        @test s == reduce(+, vh; init)
    end

    # Test with unmaterialised ranges
    for _ in 1:100
        num_elems = rand(1:1000)
        v = 1:num_elems
        s = AK.reduce(+, v, BACKEND; prefer_threads, init=Int32(0))
        vh = Array(v)
        @test s == reduce(+, vh)
    end

    # Base-compatible alias: dims=: reduces all dimensions to a scalar.
    vh_colon = rand(Int32(1):Int32(10), 3, 4, 5)
    @test AK.reduce(+, array_from_host(vh_colon); prefer_threads, init=Int32(0), dims=:) ==
        reduce(+, vh_colon; init=Int32(0), dims=:)

    vh_one = Int32[7]
    @test AK.reduce(+, array_from_host(vh_one); prefer_threads, init=Int32(10)) ==
        reduce(+, vh_one; init=Int32(10))

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.reduce(+, array_from_host(rand(Int32, 10)); init=10, bad=:kwarg)
    if !IS_CPU_BACKEND
        @test_throws ArgumentError AK.reduce(+, array_from_host(rand(Int32, 256)); prefer_threads, init=Int32(0), block_size=192)
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 10_000));
        prefer_threads,
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
        rand(Int32, 10_000);
        prefer_threads,
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
                    d = AK.reduce(+, s; prefer_threads, init=Int32(10), dims)
                    dh = Array(d)
                    @test dh == sum(sh; init=Int32(10), dims)
                    @test eltype(dh) == eltype(sum(sh; init=Int32(10), dims))
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
            s = AK.reduce(+, v; prefer_threads, init=Int32(0), dims)
            sh = Array(s)
            @test sh == sum(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; prefer_threads, init=UInt32(0), dims)
            sh = Array(s)
            @test sh == sum(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; prefer_threads, init=Float32(0), dims)
            sh = Array(s)
            @test sh ≈ sum(vh; dims)
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
            s = AK.reduce(+, v; prefer_threads, init=Int32(init), dims)
            sh = Array(s)
            @test sh == reduce(+, vh; dims, init)
        end
    end

    # Duplicate dims match Base semantics and are reduced once.
    vh_dup = rand(Int32(1):Int32(10), 3, 4, 5)
    @test Array(AK.reduce(+, array_from_host(vh_dup); prefer_threads, init=Int32(0), dims=(2,2))) ==
        sum(vh_dup; init=Int32(0), dims=(2,2))

    # min/max with dims: tests correct neutral element in partial reduction
    for dims in 1:3
        n1 = rand(1:50); n2 = rand(1:50); n3 = rand(1:50)
        vh = rand(Int32(1):Int32(100), n1, n2, n3)
        v = array_from_host(vh)
        @test Array(AK.reduce(min, v; prefer_threads, init=typemax(Int32), neutral=typemax(Int32), dims)) == minimum(vh; dims)
        @test Array(AK.reduce(max, v; prefer_threads, init=typemin(Int32), neutral=typemin(Int32), dims)) == maximum(vh; dims)
    end

    # Tuple dims support. Order and duplicates match Base semantics.
    for dims in [(1,2), (1,3), (2,3), (1,2,3), (2,1), (3,1), (2,1,2)]
        for n1 in [1, 5, 10], n2 in [1, 5, 10], n3 in [1, 5, 10]
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; prefer_threads, init=Int32(0), dims)
            sh = Array(s)
            @test sh == sum(vh; dims)
        end
    end

    # Tiled strided GPU path: contiguous kept dimensions, one strided reduce
    # dimension, and dst_size == reduce_size. The 3D case also exercises a
    # partial output tile.
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, v; prefer_threads, init=Int32(0), dims)) ==
            sum(vh; init=Int32(0), dims)
    end

    if IS_CPU_BACKEND
        # The CPU fallback should not require strided storage.
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.reduce(+, vh, BACKEND; prefer_threads, init=0, dims=(1,2))) ==
            sum(vh; init=0, dims=(1,2))
    else
        # Strided GPU sources (views, adjoints, permuted dims) take the stride-based
        # fast path over their dense parent buffer; the offset view exercises a nonzero
        # base offset. Broadcasted/lazy sources still take the generic fallback.
        vh = reshape(Int32(1):Int32(40), 5, 8)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, @view(v[:, 1:2:end]); prefer_threads, init=Int32(0), dims=2)) ==
            Base.reduce(+, @view(vh[:, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.reduce(+, @view(v[2:end, 1:2:end]); prefer_threads, init=Int32(0), dims=2)) ==
            Base.reduce(+, @view(vh[2:end, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.reduce(+, v'; prefer_threads, init=Int32(0), dims=1)) ==
            Base.reduce(+, vh'; init=Int32(0), dims=1)
        @test Array(AK.reduce(+, PermutedDimsArray(v, (2, 1)); prefer_threads, init=Int32(0), dims=1)) ==
            Base.reduce(+, PermutedDimsArray(vh, (2, 1)); init=Int32(0), dims=1)
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.reduce(+, array_from_host(rand(Int32, 10, 10)); prefer_threads, init=10, bad=:kwarg)
    if !IS_CPU_BACKEND
        @test_throws ArgumentError AK.reduce(+, array_from_host(rand(Int32, 16, 16)); prefer_threads, init=Int32(0), dims=1, block_size=192)
    end

    # Testing different settings
    AK.reduce(
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
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
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
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

    if IS_CPU_BACKEND
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
    if !IS_CPU_BACKEND
        @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 256)); prefer_threads, init=Int32(0), block_size=192)
    end
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
                    d = AK.mapreduce(-, +, s; prefer_threads, init=Int32(-10), dims)
                    dh = Array(d)
                    @test dh == mapreduce(-, +, sh; init=Int32(-10), dims)
                    @test eltype(dh) == eltype(mapreduce(-, +, sh; init=Int32(-10), dims))
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
            s = AK.mapreduce(-, +, v; prefer_threads, init=Int32(0), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; init=Int32(0), dims)
        end
    end

    function minbox(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            prefer_threads,
            init=(typemax(Float32), typemax(Float32)),
            neutral=(typemax(Float32), typemax(Float32)),
            dims,
        )
    end

    function minbox_base(s, dims)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        Base.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            init=(typemax(Float32), typemax(Float32)),
            dims,
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
            s = AK.mapreduce(-, +, v; prefer_threads, init=Int32(init), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; dims, init)
        end
    end

    # Duplicate dims match Base semantics and are reduced once.
    vh_dup = rand(Int32(1):Int32(10), 3, 4, 5)
    @test Array(AK.mapreduce(-, +, array_from_host(vh_dup); prefer_threads, init=Int32(0), dims=(2,2))) ==
        mapreduce(-, +, vh_dup; init=Int32(0), dims=(2,2))

    # Multi-input mapreduce with dimensional reductions.
    vh_ma = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_mb = rand(Int32(-10):Int32(10), 4, 5, 6)
    v_ma = array_from_host(vh_ma)
    v_mb = array_from_host(vh_mb)
    for dims in (1, 2, (1, 2), (1, 3), (1, 2, 3), (2, 2))
        @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb; prefer_threads, init=Int32(0), dims)) ==
            mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims)
    end
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb, BACKEND; prefer_threads, init=Int32(0), dims=(1, 2))) ==
        mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims=(1, 2))
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb; prefer_threads, init=Int32(0), dims=())) ==
        mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims=())
    @test Array(AK.mapreduce((x, y) -> Float32(x - y) / 3, +, v_ma, v_mb; prefer_threads, init=0f0, dims=(1, 2))) ≈
        mapreduce((x, y) -> Float32(x - y) / 3, +, vh_ma, vh_mb; init=0f0, dims=(1, 2))
    vh_typechange_nd = rand(Int32(-10):Int32(10), 4, 5)
    f_min_typechange_nd = x -> Float32(10_000_000_000 + x)
    f_max_typechange_nd = x -> Float32(-10_000_000_000 + x)
    @test Array(AK.mapreduce(f_min_typechange_nd, min, array_from_host(vh_typechange_nd); prefer_threads, init=Inf32, dims=2)) ≈
        mapreduce(f_min_typechange_nd, min, vh_typechange_nd; init=Inf32, dims=2)
    @test Array(AK.mapreduce(f_max_typechange_nd, max, array_from_host(vh_typechange_nd); prefer_threads, init=-Inf32, dims=2)) ≈
        mapreduce(f_max_typechange_nd, max, vh_typechange_nd; init=-Inf32, dims=2)

    # min/max with dims: tests correct neutral element in partial reduction
    for dims in 1:3
        n1 = rand(1:50); n2 = rand(1:50); n3 = rand(1:50)
        vh = rand(Int32(1):Int32(100), n1, n2, n3)
        v = array_from_host(vh)
        @test Array(AK.reduce(min, v; prefer_threads, init=typemax(Int32), neutral=typemax(Int32), dims)) == minimum(vh; dims)
        @test Array(AK.reduce(max, v; prefer_threads, init=typemin(Int32), neutral=typemin(Int32), dims)) == maximum(vh; dims)
    end

    # Tuple dims support. Order and duplicates match Base semantics.
    for dims in [(1,2), (1,3), (2,3), (1,2,3), (2,1), (3,1), (2,1,2)]
        for n1 in [1, 5, 10], n2 in [1, 5, 10], n3 in [1, 5, 10]
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mapreduce(-, +, v; prefer_threads, init=Int32(0), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; init=Int32(0), dims)
        end
    end

    # Tiled strided GPU path coverage for mapreduce, including a 3D case with
    # a partial output tile.
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, v; prefer_threads, init=Int32(0), dims)) ==
            mapreduce(x -> x - Int32(1), +, vh; init=Int32(0), dims)
    end

    if IS_CPU_BACKEND
        # The CPU fallback should not require strided storage.
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.mapreduce(x -> 2x, +, vh, BACKEND; prefer_threads, init=0, dims=(1,2))) ==
            mapreduce(x -> 2x, +, vh; init=0, dims=(1,2))
    else
        # Strided GPU sources (views, adjoints, permuted dims) take the stride-based
        # fast path over their dense parent buffer; the offset view exercises a nonzero
        # base offset. Broadcasted/lazy sources still take the generic fallback.
        vh = reshape(Int32(1):Int32(40), 5, 8)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, @view(v[:, 1:2:end]); prefer_threads, init=Int32(0), dims=2)) ==
            mapreduce(x -> x - Int32(1), +, @view(vh[:, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, @view(v[2:end, 1:2:end]); prefer_threads, init=Int32(0), dims=2)) ==
            mapreduce(x -> x - Int32(1), +, @view(vh[2:end, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, PermutedDimsArray(v, (2, 1)); prefer_threads, init=Int32(0), dims=1)) ==
            mapreduce(x -> x - Int32(1), +, PermutedDimsArray(vh, (2, 1)); init=Int32(0), dims=1)
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.mapreduce(-, +, array_from_host(rand(Int32, 3, 4, 5)); prefer_threads, init=10, bad=:kwarg)
    if !IS_CPU_BACKEND
        @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 16, 16)); prefer_threads, init=Int32(0), dims=1, block_size=192)
    end

    # Testing different settings
    AK.mapreduce(
        -,
        (x, y) -> x + 1,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
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
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
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
    @test AK.sum(v; prefer_threads) == sum(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.sum(v; prefer_threads) ≈ sum(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.sum(v; prefer_threads) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v; prefer_threads, dims))
            rh = sum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.sum(v; prefer_threads, block_size=64)
    @test AK.sum(v; prefer_threads, dims=:) == sum(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.sum(v; prefer_threads, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


@testset "prod" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.prod(v; prefer_threads) == prod(Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.prod(v; prefer_threads) ≈ prod(Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.sum(v; prefer_threads) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v; prefer_threads, dims))
            rh = sum(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.prod(v; prefer_threads, block_size=64)
    @test AK.prod(v; prefer_threads, dims=:) == prod(Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.prod(v; prefer_threads, bad=:kwarg)

    # The other settings are stress-tested in reduce
end


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


@testset "count" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    @test AK.count(x->x>50, v; prefer_threads) == count(x->x>50, Array(v))

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.count(x->x>0.5, v; prefer_threads) == count(x->x>0.5, Array(v))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear
            @test AK.count(x->x>0.5, v; prefer_threads) == count(x->x>0.5, vh)

            # Along dimensions
            r = Array(AK.count(x->x>0.5, v; prefer_threads, dims))
            rh = count(x->x>0.5, vh; dims)

            @test r == rh
        end
    end

    # Counting booleans directly
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Bool, num_elems))
        @test AK.count(v; prefer_threads) == count(Array(v))
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.count(x->x>0, v; prefer_threads, block_size=64)
    @test AK.count(x->x>0, v; prefer_threads, dims=:) == count(x->x>0, Array(v); dims=:)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.count(v; prefer_threads, bad=:kwarg)

    # The other settings are stress-tested in reduce
end
