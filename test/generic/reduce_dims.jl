@testset "reduce: dims" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.reduce(+, s; alg=REDUCE_ALG, init=Int32(10), dims)
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
            s = AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(0), dims)
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
            s = AK.reduce(+, v; alg=REDUCE_ALG, init=UInt32(0), dims)
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
            s = AK.reduce(+, v; alg=REDUCE_ALG, init=Float32(0), dims)
            sh = Array(s)
            @test sh ≈ sum(vh; dims)
        end
    end

    # The init value is respected
    for _ in 1:100
        for dims in 1:4
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(init), dims)
            sh = Array(s)
            @test sh == reduce(+, vh; dims, init)
        end
    end

    # Duplicate dims match Base semantics and are reduced once.
    vh_dup = rand(Int32(1):Int32(10), 3, 4, 5)
    @test Array(AK.reduce(+, array_from_host(vh_dup); alg=REDUCE_ALG, init=Int32(0), dims=(2,2))) ==
        sum(vh_dup; init=Int32(0), dims=(2,2))

    # min/max with dims: tests correct neutral element in partial reduction
    for dims in 1:3
        n1 = rand(1:50); n2 = rand(1:50); n3 = rand(1:50)
        vh = rand(Int32(1):Int32(100), n1, n2, n3)
        v = array_from_host(vh)
        @test Array(AK.reduce(min, v; alg=REDUCE_ALG, init=typemax(Int32), neutral=typemax(Int32), dims)) == minimum(vh; dims)
        @test Array(AK.reduce(max, v; alg=REDUCE_ALG, init=typemin(Int32), neutral=typemin(Int32), dims)) == maximum(vh; dims)
    end

    # Tuple dims support. Order and duplicates match Base semantics.
    for dims in [(1,2), (1,3), (2,3), (1,2,3), (2,1), (3,1), (2,1,2)]
        for n1 in [1, 5, 10], n2 in [1, 5, 10], n3 in [1, 5, 10]
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(0), dims)
            sh = Array(s)
            @test sh == sum(vh; dims)
        end
    end

    # Base also accepts iterable dims such as vectors and ranges.
    for dims in ([1,2], [1,3], [2,3], [1,2,3], [2,1], [2,1,2], Int[], Any[1,2], Int32[1,2], 1:2)
        vh = rand(Int32(1):Int32(100), 3, 4, 5)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            sum(vh; init=Int32(0), dims)
    end

    @test_throws ArgumentError AK.reduce(+, array_from_host(rand(Int32, 3, 4)); alg=REDUCE_ALG, init=Int32(0), dims=[1.0, 2.0])

    # Contiguous outputs, each reducing as many elements a stride apart as there are outputs; the
    # 3-d case's outputs do not fill whole tiles
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            sum(vh; init=Int32(0), dims)
    end

    if TEST_KERNELS
        # Columns of 4- and 8-byte elements, read with vector loads (see "reduce: launch paths")
        for T in valid_backend_eltypes(BACKEND, (Float32, Int32, UInt32, Float64, Int64, UInt64))
            vh = T <: AbstractFloat ? rand(T, 1024, 1024) : rand(T(1):T(100), 1024, 1024)
            v = array_from_host(vh)
            r = Array(AK.reduce(+, v; alg=REDUCE_ALG, init=zero(T), dims=1))
            @test T <: AbstractFloat ? r ≈ sum(vh; dims=1) : r == sum(vh; dims=1)

            wh = T <: AbstractFloat ? rand(T, 256, 32, 16) : rand(T(1):T(100), 256, 32, 16)
            w = array_from_host(wh)
            rw = Array(AK.reduce(+, w; alg=REDUCE_ALG, init=zero(T), dims=1))
            @test T <: AbstractFloat ? rw ≈ sum(wh; dims=1) : rw == sum(wh; dims=1)
        end

        # A mapping function, and neutral elements other than zero
        fh = rand(Float32, 256, 1024)
        fv = array_from_host(fh)
        @test Array(AK.mapreduce(abs2, +, fv; alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
            mapreduce(abs2, +, fh; dims=1)

        mh = rand(Int32(1):Int32(1000), 256, 1024)
        mv = array_from_host(mh)
        @test Array(AK.reduce(min, mv; alg=REDUCE_ALG, init=typemax(Int32), neutral=typemax(Int32), dims=1)) ==
            minimum(mh; dims=1)
        @test Array(AK.reduce(max, mv; alg=REDUCE_ALG, init=typemin(Int32), neutral=typemin(Int32), dims=1)) ==
            maximum(mh; dims=1)

        # Columns whose length is not a multiple of the vector width, with many and few of them
        ph = rand(Float32, 1028, 2048)
        p = array_from_host(ph)
        for rows in (1026, 1025)
            @test Array(AK.reduce(+, @view(p[1:rows, :]); alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
                sum(@view(ph[1:rows, :]); dims=1)
        end

        gh = rand(Float32, 1028, 64)
        g = array_from_host(gh)
        @test Array(AK.reduce(+, @view(g[1:1025, :]); alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
            sum(@view(gh[1:1025, :]); dims=1)

        qh = rand(Int64(1):Int64(100), 1028, 2048)
        q = array_from_host(qh)
        @test Array(AK.reduce(+, @view(q[1:1027, :]); alg=REDUCE_ALG, init=Int64(0), dims=1)) ==
            sum(@view(qh[1:1027, :]); dims=1)

        # Columns at offsets that are not multiples of the vector width, which take scalar loads
        rh = rand(Float32, 1027, 2048)
        r = array_from_host(rh)
        @test Array(AK.reduce(+, @view(r[1:1024, :]); alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
            sum(@view(rh[1:1024, :]); dims=1)

        bh = rand(Int32(1):Int32(100), 260, 1024)
        b = array_from_host(bh)
        @test Array(AK.reduce(+, @view(b[3:end, :]); alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            Base.reduce(+, @view(bh[3:end, :]); init=Int32(0), dims=1)
    end

    if !TEST_KERNELS
        # A source without strided storage
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.reduce(+, vh; alg=REDUCE_ALG, backend=BACKEND, init=0, dims=(1,2))) ==
            sum(vh; init=0, dims=(1,2))
    else
        # Strided views, adjoints and permuted dimensions of a dense array, read through its
        # buffer; the second view starts at an offset
        vh = reshape(Int32(1):Int32(40), 5, 8)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, @view(v[:, 1:2:end]); alg=REDUCE_ALG, init=Int32(0), dims=2)) ==
            Base.reduce(+, @view(vh[:, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.reduce(+, @view(v[2:end, 1:2:end]); alg=REDUCE_ALG, init=Int32(0), dims=2)) ==
            Base.reduce(+, @view(vh[2:end, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.reduce(+, v'; alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            Base.reduce(+, vh'; init=Int32(0), dims=1)
        @test Array(AK.reduce(+, PermutedDimsArray(v, (2, 1)); alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            Base.reduce(+, PermutedDimsArray(vh, (2, 1)); init=Int32(0), dims=1)
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.reduce(+, array_from_host(rand(Int32, 10, 10)); alg=REDUCE_ALG, init=10, bad=:kwarg)
    if TEST_KERNELS
        @test_throws ArgumentError AK.reduce(+, array_from_host(rand(Int32, 16, 16)); alg=reduce_alg(block_size=192), init=Int32(0), dims=1)
    end

    # Testing different settings
    AK.mapreducedim!(
        identity,
        (x, y) -> x + 1,
        array_from_host(zeros(Int32, 3, 1, 5)),
        array_from_host(rand(Int32, 3, 4, 5));
        alg=reduce_alg(block_size=64, max_tasks=10, min_elems=100),
        init=Int32(0),
        neutral=Int32(0),
    )
    AK.mapreducedim!(
        identity,
        (x, y) -> x + 1,
        array_from_host(zeros(Int32, 3, 4, 1)),
        array_from_host(rand(Int32, 3, 4, 5));
        alg=reduce_alg(block_size=64, max_tasks=16, min_elems=1000),
        init=Int32(0),
        neutral=Int32(0),
    )
end


@testset "mapreduce: dims" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.reduce
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(-100):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.mapreduce(-, +, s; alg=REDUCE_ALG, init=Int32(-10), dims)
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
            s = AK.mapreduce(-, +, v; alg=REDUCE_ALG, init=Int32(0), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; init=Int32(0), dims)
        end
    end

    function minbox(s, dims; alg=REDUCE_ALG)
        # Extract coordinates into tuple and reduce to find dimensionwise minima
        AK.mapreduce(
            p -> (p.x, p.y),
            (a, b) -> (a[1] < b[1] ? a[1] : b[1], a[2] < b[2] ? a[2] : b[2]),
            s;
            alg,
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
            mgpu = minbox(v, dims; alg=REDUCE_ALG)

            vh = Array(v)
            mcpu = minbox(vh, dims; alg=AK.Auto())
            mbase = minbox_base(vh, dims)

            @test eltype(mgpu) === eltype(mcpu) === eltype(mbase)
            @test all([
                (mgpu_red[1] ≈ mcpu[i][1] ≈ mbase[i][1]) && (mgpu_red[2] ≈ mcpu[i][2] ≈ mbase[i][2])
                for (i, mgpu_red) in enumerate(Array(mgpu))
            ])
        end
    end

    # The init value is respected
    for _ in 1:100
        for dims in 1:4
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(-100):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.mapreduce(-, +, v; alg=REDUCE_ALG, init=Int32(init), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; dims, init)
        end
    end

    # Duplicate dims match Base semantics and are reduced once.
    vh_dup = rand(Int32(1):Int32(10), 3, 4, 5)
    @test Array(AK.mapreduce(-, +, array_from_host(vh_dup); alg=REDUCE_ALG, init=Int32(0), dims=(2,2))) ==
        mapreduce(-, +, vh_dup; init=Int32(0), dims=(2,2))

    # Multi-input mapreduce with dimensional reductions.
    vh_ma = rand(Int32(-10):Int32(10), 4, 5, 6)
    vh_mb = rand(Int32(-10):Int32(10), 4, 5, 6)
    v_ma = array_from_host(vh_ma)
    v_mb = array_from_host(vh_mb)
    for dims in (1, 2, (1, 2), (1, 3), (1, 2, 3), (2, 2))
        @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims)
    end
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb; alg=REDUCE_ALG, backend=BACKEND, init=Int32(0), dims=(1, 2))) ==
        mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims=(1, 2))
    @test Array(AK.mapreduce((x, y) -> x * y, +, v_ma, v_mb; alg=REDUCE_ALG, init=Int32(0), dims=())) ==
        mapreduce((x, y) -> x * y, +, vh_ma, vh_mb; init=Int32(0), dims=())
    @test Array(AK.mapreduce((x, y) -> Float32(x - y) / 3, +, v_ma, v_mb; alg=REDUCE_ALG, init=0f0, dims=(1, 2))) ≈
        mapreduce((x, y) -> Float32(x - y) / 3, +, vh_ma, vh_mb; init=0f0, dims=(1, 2))
    vh_typechange_nd = rand(Int32(-10):Int32(10), 4, 5)
    f_min_typechange_nd = x -> Float32(10_000_000_000 + x)
    f_max_typechange_nd = x -> Float32(-10_000_000_000 + x)
    @test Array(AK.mapreduce(f_min_typechange_nd, min, array_from_host(vh_typechange_nd); alg=REDUCE_ALG, init=Inf32, dims=2)) ≈
        mapreduce(f_min_typechange_nd, min, vh_typechange_nd; init=Inf32, dims=2)
    @test Array(AK.mapreduce(f_max_typechange_nd, max, array_from_host(vh_typechange_nd); alg=REDUCE_ALG, init=-Inf32, dims=2)) ≈
        mapreduce(f_max_typechange_nd, max, vh_typechange_nd; init=-Inf32, dims=2)

    # min/max with dims: tests correct neutral element in partial reduction
    for dims in 1:3
        n1 = rand(1:50); n2 = rand(1:50); n3 = rand(1:50)
        vh = rand(Int32(1):Int32(100), n1, n2, n3)
        v = array_from_host(vh)
        @test Array(AK.reduce(min, v; alg=REDUCE_ALG, init=typemax(Int32), neutral=typemax(Int32), dims)) == minimum(vh; dims)
        @test Array(AK.reduce(max, v; alg=REDUCE_ALG, init=typemin(Int32), neutral=typemin(Int32), dims)) == maximum(vh; dims)
    end

    # Tuple dims support. Order and duplicates match Base semantics.
    for dims in [(1,2), (1,3), (2,3), (1,2,3), (2,1), (3,1), (2,1,2)]
        for n1 in [1, 5, 10], n2 in [1, 5, 10], n3 in [1, 5, 10]
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mapreduce(-, +, v; alg=REDUCE_ALG, init=Int32(0), dims)
            sh = Array(s)
            @test sh == mapreduce(-, +, vh; init=Int32(0), dims)
        end
    end

    # Base also accepts iterable dims such as vectors and ranges.
    for dims in ([1,2], [1,3], [2,3], [1,2,3], [2,1], [2,1,2], Int[], Any[1,2], Int32[1,2], 1:2)
        vh = rand(Int32(1):Int32(100), 3, 4, 5)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(-, +, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            mapreduce(-, +, vh; init=Int32(0), dims)
    end

    @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 3, 4)); alg=REDUCE_ALG, init=Int32(0), dims=[1.0, 2.0])

    # Contiguous outputs, each reducing as many elements a stride apart as there are outputs; the
    # 3-d case's outputs do not fill whole tiles
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            mapreduce(x -> x - Int32(1), +, vh; init=Int32(0), dims)
    end

    if !TEST_KERNELS
        # A source without strided storage
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.mapreduce(x -> 2x, +, vh; alg=REDUCE_ALG, backend=BACKEND, init=0, dims=(1,2))) ==
            mapreduce(x -> 2x, +, vh; init=0, dims=(1,2))
    else
        # Strided views, adjoints and permuted dimensions of a dense array, read through its
        # buffer; the second view starts at an offset
        vh = reshape(Int32(1):Int32(40), 5, 8)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, @view(v[:, 1:2:end]); alg=REDUCE_ALG, init=Int32(0), dims=2)) ==
            mapreduce(x -> x - Int32(1), +, @view(vh[:, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, @view(v[2:end, 1:2:end]); alg=REDUCE_ALG, init=Int32(0), dims=2)) ==
            mapreduce(x -> x - Int32(1), +, @view(vh[2:end, 1:2:end]); init=Int32(0), dims=2)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, PermutedDimsArray(v, (2, 1)); alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            mapreduce(x -> x - Int32(1), +, PermutedDimsArray(vh, (2, 1)); init=Int32(0), dims=1)
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.mapreduce(-, +, array_from_host(rand(Int32, 3, 4, 5)); alg=REDUCE_ALG, init=10, bad=:kwarg)
    if TEST_KERNELS
        @test_throws ArgumentError AK.mapreduce(-, +, array_from_host(rand(Int32, 16, 16)); alg=reduce_alg(block_size=192), init=Int32(0), dims=1)
    end

    # Testing different settings
    AK.mapreducedim!(
        -,
        (x, y) -> x + 1,
        array_from_host(zeros(Int32, 3, 1, 5)),
        array_from_host(rand(Int32, 3, 4, 5));
        alg=reduce_alg(block_size=64, max_tasks=10, min_elems=100),
        init=Int32(0),
        neutral=Int32(0),
    )
    AK.mapreducedim!(
        -,
        (x, y) -> x + 1,
        array_from_host(zeros(Int32, 3, 4, 1)),
        array_from_host(rand(Int32, 3, 4, 5));
        alg=reduce_alg(block_size=64, max_tasks=16, min_elems=1000),
        init=Int32(0),
        neutral=Int32(0),
    )
end


@testset "reduce: block sizes" begin
    # The block-level tree reduction must cover every power-of-two group size the device allows;
    # each shape below reaches a different kernel: the 1D block reduction, the grid-strided
    # one-block-per-output reduction (contiguous and strided sources), the multi-block
    # reduction with its second pass, and the column reduction with its second pass.
    Random.seed!(0)
    vh = rand(Int32(1):Int32(100), 100_000)
    v = array_from_host(vh)
    shapes = (((3000, 40), 1), ((40, 3000), 2), ((20_000, 4), 1), ((64, 8192), 2),
              ((10, 30_000), 2), ((3000, 4000), 2))
    mats = [(array_from_host(rand(Int32(1):Int32(100), sz)), dims) for (sz, dims) in shapes]
    block_sizes = TEST_KERNELS ? filter(<=(MAX_BLOCK_SIZE), 2 .^ (0:10)) : [nothing]
    for block_size in block_sizes
        alg = reduce_alg(; block_size)
        @test AK.reduce(+, v; init=Int32(0), alg) == sum(vh)
        for (m, dims) in mats
            @test Array(AK.reduce(+, m; init=Int32(0), dims, alg)) == sum(Array(m); dims)
        end
    end
end


@testset "reduce: runs of consecutive elements" begin
    # Outputs that each reduce a run of consecutive elements, read by groups of threads: runs of
    # several lengths (with few outputs, taking more threads per run), and a dimension of size 1
    # among the kept ones
    Random.seed!(0)
    alg = REDUCE_ALG
    for (sz, dims) in (((16, 3000), 1), ((512, 1000), 1), ((513, 47), 1), ((4096, 33), 1),
                       ((64, 1, 300), 1), ((4, 8, 3000), (1, 2)))
        h = rand(Int32(-9):Int32(9), sz)
        a = array_from_host(h)
        rsz = length.(Base.reduced_indices(axes(h), dims))
        # `init` applied once, overwriting, and folding into `R`
        R = array_from_host(zeros(Int32, rsz))
        AK.mapreducedim!(identity, +, R, a; init=Int32(2), alg)
        @test Array(R) == sum(h; dims) .+ 2
        AK.mapreducedim!(identity, max, R, a; overwrite=true, alg)
        @test Array(R) == maximum(h; dims)
        R0 = rand(Int32(-9):Int32(9), rsz)
        R = array_from_host(R0)
        AK.mapreducedim!(identity, +, R, a; alg)
        @test Array(R) == R0 .+ sum(h; dims)
        # A mapping function and an operator without a neutral element
        R = array_from_host(zeros(Int32, rsz))
        AK.mapreducedim!(abs2, (x, y) -> x + y, R, a; init=Int32(0), alg)
        @test Array(R) == sum(abs2, h; dims)
        # A view at an offset, and a lazy source
        hp = rand(Int32(-9):Int32(9), prod(sz) + 7)
        v = reshape(view(array_from_host(hp), 8:prod(sz) + 7), sz)
        AK.mapreducedim!(identity, +, R, v; init=Int32(0), alg)
        @test Array(R) == sum(reshape(hp[8:end], sz); dims)
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(*, a, Int32(2)))
        AK.mapreducedim!(identity, +, R, bc; init=Int32(0), alg)
        @test Array(R) == 2 .* sum(h; dims)
    end
    # Runs inside a strided view, whose kept dimensions are not adjacent in memory
    h = rand(Int32(-9):Int32(9), 100, 6, 70)
    v = view(array_from_host(h), :, 2:5, :)
    @test Array(AK.reduce(+, v; dims=1, init=Int32(0), alg)) == sum(h[:, 2:5, :]; dims=1)
    # Blocks too small for many threads per run
    if TEST_KERNELS
        h = rand(Int32(-9):Int32(9), 512, 100)
        for block_size in (1, 2, 8, 32)
            r = AK.reduce(+, array_from_host(h); dims=1, init=Int32(0),
                          alg=AK.BlockReduce(; block_size))
            @test Array(r) == sum(h; dims=1)
        end
    end
end


@testset "reduce: strided slices" begin
    # Few contiguous outputs, each reducing a long strided row, take the column reduction in two
    # passes (with enough elements for every backend's threshold): a tile of 8 outputs, and two
    # tiles of 32 outputs the second of which is partial
    Random.seed!(0)
    alg = REDUCE_ALG
    for (m, n) in ((8, 600_000), (40, 110_000))
        h = rand(Int8(-9):Int8(9), m, n)
        a = array_from_host(h)
        s = sum(Int.(h); dims=2)
        # `init` applied once, folding into `R`, into an `R` narrower than the accumulator
        R = array_from_host(fill(Int16(1), m))
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, alg)
        @test Array(R) == vec(s) .+ 1
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, init=Int16(7), alg)
        @test Array(R) == vec(s) .+ 7
        # A workspace, used twice
        ws = AK.workspace(AK.mapreducedim!, identity, +, R, a; acctype=Int32, alg)
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, overwrite=true, alg, workspace=ws)
        @test Array(R) == vec(s)
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, alg, workspace=ws)
        @test Array(R) == 2 .* vec(s)
        # Without a neutral element
        @test Array(AK.reduce((x, y) -> max(x, y), a; dims=2, alg)) == maximum(h; dims=2)
        # A strided view at an offset
        p = array_from_host(rand(Int8(-9):Int8(9), m + 2, n + 3))
        v = view(p, 2:m + 1, 4:n + 3)
        @test Array(AK.sum(v; dims=2, alg)) == sum(Int.(Array(p)[2:m + 1, 4:n + 3]); dims=2)
    end
    # More outputs than elements each reduces: the tiled kernel, unless tiles of 32 outputs
    # launch enough blocks (the columns kernel in one pass), and one thread per output where a row
    # of outputs fills the device
    for (m, n, A) in ((4096, 1024, Int32), (8192, 300, Int32), (8192, 300, Int64))
        h = rand(A(-9):A(9), m, n)
        a = array_from_host(h)
        r = AK.reduce(+, a; dims=2, init=zero(A), alg)
        @test eltype(r) === A && Array(r) == sum(h; dims=2)
        R = array_from_host(ones(A, m))
        AK.mapreducedim!(identity, +, R, a; alg)
        @test Array(R) == vec(sum(h; dims=2)) .+ 1
    end
    # Runs of contiguous outputs at the offsets of other kept dimensions: the middle dimension of
    # 3-d and 4-d arrays, and a view at an offset whose kept dimensions are not adjacent in memory
    h = rand(Int32(-9):Int32(9), 50, 30_000, 7)
    for (a, ha) in ((array_from_host(rand(Int32(-9):Int32(9), 96, 7000, 10)), nothing),
                    (array_from_host(rand(Int32(-9):Int32(9), 40, 30_000, 3, 2)), nothing),
                    (view(array_from_host(h), 3:42, :, 1:2:7), h[3:42, :, 1:2:7]))
        ha = something(ha, Array(a))
        @test Array(AK.reduce(+, a; dims=2, init=Int32(0), alg)) == sum(ha; dims=2)
    end
end


@testset "reduce: dims, into a single output" begin
    # Along `dims`, a reduction into one output of a contiguous or fused source takes the
    # whole-array kernels, finishing on the device; the sizes span one tile, one pass and two
    Random.seed!(0)
    alg = REDUCE_ALG
    for n in (2, 1000, 50_000, 3_000_000)
        h = rand(Int32(-9):Int32(9), n)
        hm = reshape(h, :, 2)
        v = array_from_host(h)
        m = array_from_host(hm)
        # `init` applied once, folding into `R`'s value, overwriting it
        R = array_from_host(Int32[5])
        AK.mapreducedim!(identity, +, R, v; alg)
        @test Array(R) == [5 + sum(h)]
        AK.mapreducedim!(identity, +, R, v; init=Int32(1), alg)
        @test Array(R) == [1 + sum(h)]
        AK.mapreducedim!(identity, +, R, m; overwrite=true, alg)
        @test Array(R) == [sum(h)]
        # The accumulator type, and an operator without a neutral element
        r = AK.sum(m; dims=(1, 2), alg)
        @test eltype(r) === Int && Array(r) == sum(hm; dims=(1, 2))
        @test Array(AK.reduce((a, b) -> max(a, b), m; dims=(1, 2), alg)) == fill(maximum(h), 1, 1)
        # A contiguous view at an offset, and a fused source
        R = array_from_host(Int32[0])
        AK.mapreducedim!(identity, +, R, view(m, :, 2); alg)
        @test Array(R) == [sum(view(hm, :, 2))]
        @test Array(AK.mapreduce(*, +, m, m; dims=(1, 2), alg)) == fill(sum(h .* h), 1, 1)
        # Partial results in the accumulator type, which only the result must fit `R`'s
        n == 2 && continue
        h8 = Int8[fill(Int8(100), n ÷ 2); fill(Int8(-100), n ÷ 2)]
        h8[end] = -95
        v8 = array_from_host(h8)
        R = array_from_host(Int8[0])
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, overwrite=true, alg)
        @test Array(R) == Int8[5]
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, init=Int8(-10), alg)
        @test Array(R) == Int8[-5]
        # A workspace, used twice
        R = array_from_host(Int8[0])
        ws = AK.workspace(AK.mapreducedim!, identity, +, R, v8; acctype=Int32, alg)
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, alg, workspace=ws)
        @test Array(R) == Int8[5]
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, alg, workspace=ws)
        @test Array(R) == Int8[10]
    end
end


# An associative, non-commutative operator on a primitive type, which the kernels can load as
# vectors: affine maps `x -> a*x + b` of `H`-bit integers packed into twice as many bits, composed
# first to last. Odd `a`s keep the maps invertible.
affine_half(::Type{UInt32}) = UInt16
affine_half(::Type{UInt64}) = UInt32
function affine_compose(p::T, q::T) where {T <: Union{UInt32, UInt64}}
    H = affine_half(T)
    n = 4 * sizeof(T)
    a1, b1, a2, b2 = p % H, (p >> n) % H, q % H, (q >> n) % H
    return T(a2 * a1) | T(a2 * b1 + b2) << n
end
affine_rand(::Type{T}) where {T} = rand(T) | one(T)

# Reductions along `dims` in element order whose launch shapes, when commutative, are every
# shape (see "reduce: launch paths"): one thread per output (for those taking one, and short runs
# of consecutive elements), the columns kernel (in one pass or two, for columns and tiles of
# strided slices), and a block per output, or several (for long runs, few outputs, and the
# other shapes); also non-adjacent dimensions
const ORDERED_CASES = (
    ((3, 2000), 1), ((2000, 3), 1), ((2000, 3), 2), ((40, 5, 30), 1), ((40, 5, 30), 2),
    ((40, 5, 30), 3), ((40, 5, 30), (1, 3)), ((9, 4, 3000), (2, 3)), ((8, 300_000), 2),
    ((40, 70_000), 2), ((256, 256), 2), ((300, 300), 2), ((3000, 300), 1), ((300_000, 2), 1),
    ((70_000, 40), 1), ((64, 4000), 1), ((512, 40), 1), ((1, 5000, 2), 2), ((40, 3000, 30), 2))

@testset "reduce: non-commutative" begin
    Random.seed!(0)

    # Whole arrays: one tile, one block, several tiles per block, and several blocks (also with
    # small tiles, so that blocks take many tiles each); with and without a neutral element
    for (alg, dalg) in ((REDUCE_ALG, REDUCE_ALG),
                        (reduce_alg(block_size=32, items_per_thread=2, max_tasks=4),
                         reduce_alg(block_size=32, max_tasks=4))),
        neutral in (nothing, I2),
        n in (2, 3, 100, 4097, 70_000, 300_001)
        xh = [randmat2() for _ in 1:n]
        x = array_from_host(xh)
        init = randmat2()
        @test AK.reduce(matmul2, x; commutative=false, neutral, alg) == foldl(matmul2, xh)
        @test AK.reduce(matmul2, x; commutative=false, neutral, init, alg) ==
              foldl(matmul2, xh; init)
        # into a single output on the device, folding into its value
        R = array_from_host([init])
        AK.mapreducedim!(identity, matmul2, R, x; commutative=false, neutral, alg=dalg)
        @test Array(R) == [foldl(matmul2, xh; init)]
    end
    # Finishing on the host, from the source or from the first pass's partial results
    xh = [randmat2() for _ in 1:20_000]
    x = array_from_host(xh)
    for switch_below in (30_000, 100)
        alg = reduce_alg(; block_size=32, items_per_thread=2, switch_below)
        @test AK.reduce(matmul2, x; commutative=false, alg) == foldl(matmul2, xh)
    end
    # Sources read with vector loads, contiguous at an offset, strided and fused ones
    for T in (UInt32, UInt64)
        h = [affine_rand(T) for _ in 1:1_000_000]
        v = array_from_host(h)
        for r in (1:1_000_000, 3:999_998, 2:2:999_000)
            @test AK.reduce(affine_compose, view(v, r); commutative=false, alg=REDUCE_ALG) ==
                  foldl(affine_compose, view(h, r))
        end
        @test AK.mapreduce(xor, affine_compose, v, v .+ one(T); commutative=false,
                           alg=REDUCE_ALG) == foldl(affine_compose, xor.(h, h .+ one(T)))
    end
    # A workspace, used twice
    ws = AK.workspace(AK.reduce, matmul2, x; commutative=false, alg=REDUCE_ALG)
    for _ in 1:2
        @test AK.reduce(matmul2, x; commutative=false, alg=REDUCE_ALG, workspace=ws) ==
              foldl(matmul2, xh)
    end

    # Along `dims`, the inputs of `ORDERED_CASES`; with and without a neutral element
    for (sz, dims) in ORDERED_CASES
        xh = [randmat2() for _ in CartesianIndices(sz)]
        x = array_from_host(xh)
        ref = reduce(matmul2, xh; dims, init=I2)
        alg = REDUCE_ALG
        for neutral in (nothing, I2)
            @test Array(AK.reduce(matmul2, x; dims, commutative=false, neutral, alg)) == ref
            # `init` applied once, folding into `R`, overwriting it
            init = randmat2()
            R0 = [randmat2() for _ in CartesianIndices(ref)]
            R = array_from_host(R0)
            AK.mapreducedim!(identity, matmul2, R, x; commutative=false, neutral, alg)
            @test Array(R) == matmul2.(R0, ref)
            AK.mapreducedim!(identity, matmul2, R, x; commutative=false, neutral, init, alg)
            @test Array(R) == matmul2.(Ref(init), ref)
            AK.mapreducedim!(identity, matmul2, R, x; commutative=false, neutral,
                             overwrite=true, alg)
            @test Array(R) == ref
        end
    end
    # Square tiles in blocks too small for the columns kernel
    xh = [randmat2() for _ in 1:256, _ in 1:256]
    @test Array(AK.reduce(matmul2, array_from_host(xh); dims=2, commutative=false,
                          alg=reduce_alg(block_size=8))) ==
          reduce(matmul2, xh; dims=2, init=I2)
    # Sources read with vector loads (contiguous, at an offset), strided views and lazy ones;
    # few outputs of four-byte elements, whose groups outnumber the tiles
    for T in (UInt32, UInt64)
        hm = [affine_rand(T) for _ in 1:4096, _ in 1:300]
        m = array_from_host(hm)
        for (src, hsrc) in ((m, hm), (view(m, 2:4095, :), view(hm, 2:4095, :)),
                            (view(m, 1:2:4096, 2:2:300), view(hm, 1:2:4096, 2:2:300)),
                            (LazyProduct(m, m), hm .* hm)), dims in (1, 2)
            @test Array(AK.reduce(affine_compose, src; dims, commutative=false,
                                  alg=REDUCE_ALG)) ==
                  reduce(affine_compose, hsrc; dims, init=T(1))
        end
        hm = [affine_rand(T) for _ in 1:5000, _ in 1:2]
        @test Array(AK.reduce(affine_compose, array_from_host(hm); dims=1, commutative=false,
                              alg=REDUCE_ALG)) == reduce(affine_compose, hm; dims=1, init=T(1))
    end
end


# Correct results do not show which kernel ran: the inputs that the tests above give to reach
# each launch shape and vector-load path reach them with this backend's tuning
if TEST_KERNELS
@testset "reduce: launch paths" begin
    # The launch shape of a reduction of `x` along `dims` with operator `op`, as its setup
    # chooses it
    launch_shape(x, dims, op; commutative=true) =
        AK.mapreduce_setup(identity, op, x, BACKEND, AK.NoInit(), nothing, nothing, dims,
                           REDUCE_ALG; commutative).launch.shape
    # ... as a name, which tells grid-striding blocks and columns in order apart
    function shape_name(x, dims, op; commutative=true)
        shape = launch_shape(x, dims, op; commutative)
        dst_size = length(x) ÷ prod(size(x, d) for d in dims)
        shape isa AK.ByBlock && shape.blocks < dst_size && return :ByBlockStrided
        shape isa AK.Columns && shape.ordered && return :ColumnsOrdered
        return nameof(typeof(shape))
    end
    # The width of the vector loads of the reduction of `x` along `dims` (0 for scalar loads)
    function vector_width(x, dims)
        rdims = AK.reduced_dims(dims, ndims(x))
        layout = AK.strided_layout(x)
        len = prod(size(x, d) for d in rdims)
        block_size = AK.resolve_reduce(REDUCE_ALG, BACKEND, eltype(x), rdims).block_size
        AK.with_segments(AK.merge_dims(size(x), layout.strides, rdims)) do rstrides, _, ostrides, _
            AK.vector_width(layout.buffer, layout.offset, ostrides, rstrides, len, block_size)
        end
    end

    # "reduce: runs of consecutive elements"
    @test launch_shape(Array{Int32}(undef, 512, 1000), 1, +) isa AK.ByLanes
    # "reduce: strided slices", in two passes and in one
    for sz in ((8, 600_000), (40, 110_000))
        shape = launch_shape(Array{Int8}(undef, sz), 2, Base.add_sum)
        @test shape isa AK.Columns && shape.groups > 1
    end
    @test launch_shape(Array{Int32}(undef, 4096, 1024), 2, +) isa AK.Tiled
    @test launch_shape(Array{Int32}(undef, 8192, 300), 2, +) == AK.Columns(32, 1, false)
    @test launch_shape(Array{Int64}(undef, 8192, 300), 2, +) isa AK.ByThread
    for x in (Array{Int32}(undef, 96, 7000, 10), Array{Int32}(undef, 40, 30_000, 3, 2),
              view(Array{Int32}(undef, 50, 30_000, 7), 3:42, :, 1:2:7))
        @test launch_shape(x, 2, +) isa AK.Columns
    end

    # "reduce: non-commutative": every shape in order, and every shape of the commutative
    # reductions with the default settings (tunings can rule some out)
    shapes = Set(shape_name(Array{Mat2}(undef, sz), dims, matmul2) =>
                 shape_name(Array{Mat2}(undef, sz), dims, matmul2; commutative=false)
                 for (sz, dims) in ORDERED_CASES)
    @test Set(last.(shapes)) == Set((:ByThread, :ColumnsOrdered, :Ordered))
    t = AK.reduce_tuning(BACKEND, Mat2)
    if AK.resolve_reduce(REDUCE_ALG, BACKEND, Mat2, 1).block_size == 256 &&
       t.columns_min_elements <= 1 << 18
        required = Set((:ByThread => :ByThread,
                        :Columns => :ColumnsOrdered, :Tiled => :ColumnsOrdered,
                        :ByBlockStrided => :Ordered, :ByBlock => :Ordered,
                        :ByBlocks => :Ordered))
        t.segments_short_runs && push!(required, :ByLanes => :ByThread)
        # (the (512, 40) case has 256 bytes per lane)
        t.segments_lane_bytes >= 256 && push!(required, :ByLanes => :Ordered)
        @test shapes ⊇ required
    end

    # "reduce: dims": columns of aligned 4- and 8-byte elements take vector loads, those at other
    # offsets scalar ones
    for T in valid_backend_eltypes(BACKEND, (Float32, Int32, UInt32, Float64, Int64, UInt64))
        @test vector_width(array_from_host(zeros(T, 1024, 1024)), 1) == 16 ÷ sizeof(T)
    end
    @test vector_width(view(array_from_host(zeros(Float32, 1027, 2048)), 1:1024, :), 1) == 0
    @test vector_width(view(array_from_host(zeros(Int32, 260, 1024)), 3:260, :), 1) == 0
    # Buffers that are not 16-byte aligned take scalar loads
    storage = Vector{UInt8}(undef, 64)
    GC.@preserve storage begin
        offset = Int(mod(-UInt(pointer(storage)), 16)) + 4
        misaligned = unsafe_wrap(Array, Ptr{Float32}(pointer(storage) + offset), 4)
        @test vector_width(misaligned, 1) == 0
    end
end
end
