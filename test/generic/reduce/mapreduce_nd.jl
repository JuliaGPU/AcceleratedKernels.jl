struct Point
    x::Float32
    y::Float32
end
# Only for backend-agnostic initialisation with KernelAbstractions.zero
Base.zero(::Type{Point}) = Point(0.0f0, 0.0f0)
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

    function minbox(s, dims; prefer_threads)
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
            mgpu = minbox(v, dims; prefer_threads)

            vh = Array(v)
            mcpu = minbox(vh, dims; prefer_threads=true)
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

    # Base also accepts iterable dims such as vectors and ranges.
    for dims in ([1,2], [1,3], [2,3], [1,2,3], [2,1], [2,1,2], Int[], Any[1,2], Int32[1,2], 1:2)
        vh = rand(Int32(1):Int32(100), 3, 4, 5)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(-, +, v; prefer_threads, init=Int32(0), dims)) ==
            mapreduce(-, +, vh; init=Int32(0), dims)
    end

    @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 3, 4)); prefer_threads, init=Int32(0), dims=[1.0, 2.0])

    # Tiled strided GPU path coverage for mapreduce, including a 3D case with
    # a partial output tile.
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, v; prefer_threads, init=Int32(0), dims)) ==
            mapreduce(x -> x - Int32(1), +, vh; init=Int32(0), dims)
    end

    if prefer_threads
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
    if !prefer_threads
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
