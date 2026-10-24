# Tests that do not choose an algorithm use `REDUCE_ALG`: `Auto()`, except in the `--cpu-ka`
# configuration, whose point is to run AK's kernels on the host backend. `reduce_alg` builds an
# explicitly tuned algorithm for the configuration from both kinds of settings.
REDUCE_ALG = HOST_KERNELS ? AK.BlockReduce() : AK.Auto()
reduce_alg(; block_size=nothing, items_per_thread=nothing, switch_below=nothing,
           max_tasks=nothing, min_elems=nothing) =
    TEST_KERNELS ? AK.BlockReduce(; block_size, items_per_thread, switch_below) :
                   AK.CPUThreads.Partitioned(; max_tasks, min_elems)

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

    # Ensuring that the init value is respected
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


@testset "reduce_nd" begin
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

    # Ensuring that the init value is respected
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

    # Tiled strided GPU path: contiguous kept dimensions, one strided reduce
    # dimension, and dst_size == reduce_size. The 3D case also exercises a
    # partial output tile.
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.reduce(+, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            sum(vh; init=Int32(0), dims)
    end

    if TEST_KERNELS
        # Aligned stride-1 reductions use 128-bit loads for 4- and 8-byte elements.
        for T in valid_backend_eltypes(BACKEND, (Float32, Int32, UInt32, Float64, Int64, UInt64))
            vh = T <: AbstractFloat ? rand(T, 1024, 1024) : rand(T(1):T(100), 1024, 1024)
            v = array_from_host(vh)
            @test AK.vector_width(v, 0, (1024,), (1,), 1024, 256) ==
                16 ÷ sizeof(T)
            r = Array(AK.reduce(+, v; alg=REDUCE_ALG, init=zero(T), dims=1))
            @test T <: AbstractFloat ? r ≈ sum(vh; dims=1) : r == sum(vh; dims=1)

            wh = T <: AbstractFloat ? rand(T, 256, 32, 16) : rand(T(1):T(100), 256, 32, 16)
            w = array_from_host(wh)
            rw = Array(AK.reduce(+, w; alg=REDUCE_ALG, init=zero(T), dims=1))
            @test T <: AbstractFloat ? rw ≈ sum(wh; dims=1) : rw == sum(wh; dims=1)
        end

        # Apply the map lane-wise and preserve non-additive neutral elements.
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

        # Exercise scalar tails in both by-block dispatch branches.
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

        # Misaligned rows and offsets fall back to scalar loads.
        rh = rand(Float32, 1027, 2048)
        r = array_from_host(rh)
        @test AK.vector_width(r, 0, (1027,), (1,), 1024, 256) == 0
        @test Array(AK.reduce(+, @view(r[1:1024, :]); alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
            sum(@view(rh[1:1024, :]); dims=1)

        bh = rand(Int32(1):Int32(100), 260, 1024)
        b = array_from_host(bh)
        @test AK.vector_width(b, 2, (260,), (1,), 258, 256) == 0
        @test Array(AK.reduce(+, @view(b[3:end, :]); alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            Base.reduce(+, @view(bh[3:end, :]); init=Int32(0), dims=1)
    end

    if !TEST_KERNELS
        storage = Vector{UInt8}(undef, 64)
        GC.@preserve storage begin
            offset = Int(mod(-UInt(pointer(storage)), 16)) + 4
            misaligned = unsafe_wrap(Array, Ptr{Float32}(pointer(storage) + offset), 4)
            @test AK.vector_width(misaligned, 0, (4,), (1,), 4, 256) == 0
        end

        # The CPU fallback should not require strided storage.
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.reduce(+, vh; alg=REDUCE_ALG, backend=BACKEND, init=0, dims=(1,2))) ==
            sum(vh; init=0, dims=(1,2))
    else
        # Strided GPU sources (views, adjoints, permuted dims) take the stride-based
        # fast path over their dense parent buffer; the offset view exercises a nonzero
        # base offset.
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


@testset "mapreduce_1d" begin
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

    # Ensuring that the init value is respected
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


@testset "mapreduce_nd" begin
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

    # Ensuring that the init value is respected
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

    # Tiled strided GPU path coverage for mapreduce, including a 3D case with
    # a partial output tile.
    for (shape, dims) in (((512, 512), 2), ((20, 13, 260), 3))
        vh = rand(Int32(1):Int32(3), shape...)
        v = array_from_host(vh)
        @test Array(AK.mapreduce(x -> x - Int32(1), +, v; alg=REDUCE_ALG, init=Int32(0), dims)) ==
            mapreduce(x -> x - Int32(1), +, vh; init=Int32(0), dims)
    end

    if !TEST_KERNELS
        # The CPU fallback should not require strided storage.
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.mapreduce(x -> 2x, +, vh; alg=REDUCE_ALG, backend=BACKEND, init=0, dims=(1,2))) ==
            mapreduce(x -> 2x, +, vh; init=0, dims=(1,2))
    else
        # Strided GPU sources (views, adjoints, permuted dims) take the stride-based
        # fast path over their dense parent buffer; the offset view exercises a nonzero
        # base offset.
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


@testset "reduce block sizes" begin
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
    # The column reduction in one pass, which only more outputs than a test affords choose
    if TEST_KERNELS
        h = rand(Int32(1):Int32(100), 64, 1000)
        R = array_from_host(fill(Int32(1), 64))
        AK.launch_shape!(
            Val(:columns), 1, identity, +, R, array_from_host(h), BACKEND, 256, 256;
            init=AK.FoldInto(), seed=Int32(0), bufs=(;), base_offset=0, reduce_strides=(64,),
            reduce_sizes=(1000,), outer_strides=(1,), outer_sizes=(64,), dst_size=64,
            reduce_size=1000, dense_buffer=true, items_per_thread=4)
        @test Array(R) == vec(sum(h; dims=2)) .+ 1
    end
end


@testset "segment reductions" begin
    # Outputs that each reduce a run of consecutive elements, read by groups of threads: runs of
    # several lengths (with few outputs, taking more threads per run), and a dimension of size 1
    # among the kept ones
    Random.seed!(0)
    alg = REDUCE_ALG
    if TEST_KERNELS
        segs = AK.merge_dims((512, 1000), (1, 512), (1,))
        @test AK.reduce_shape(1000, 512, segs, 256, AK.reduce_tuning(BACKEND, Int32),
                              Int32)[1] === :by_lanes
    end
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


@testset "column reductions" begin
    # Few contiguous outputs, each reducing a long strided row, take the column reduction in two
    # passes (with enough elements for every backend's threshold): a tile of 8 outputs, and two
    # tiles of 32 outputs the second of which is partial
    Random.seed!(0)
    alg = REDUCE_ALG
    for (m, n) in ((8, 600_000), (40, 110_000))
        h = rand(Int8(-9):Int8(9), m, n)
        a = array_from_host(h)
        s = sum(Int.(h); dims=2)
        if TEST_KERNELS
            segs = AK.merge_dims((m, n), (1, m), (2,))
            @test AK.reduce_shape(m, n, segs, 256, AK.reduce_tuning(BACKEND, Int), Int)[1] ===
                  :columns
        end
        # `init` applied once, folding into `R`, into an `R` narrower than the accumulator
        R = array_from_host(fill(Int16(1), m))
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, alg)
        @test Array(R) == vec(s) .+ 1
        AK.mapreducedim!(identity, +, R, a; acctype=Int32, init=Int16(7), alg)
        @test Array(R) == vec(s) .+ 7
        # ... with a workspace, used twice
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
    # (the shape the reduction of `x` along `dims` with accumulator type `A` takes)
    shape_of(x, dims, A) = AK.reduce_shape(
        length(x) ÷ prod(size(x, d) for d in dims), prod(size(x, d) for d in dims),
        AK.merge_dims(size(x), AK.source_layout(x).strides, dims),
        AK.resolve_reduce(alg, BACKEND, A, dims).block_size, AK.reduce_tuning(BACKEND, A), A)[1]
    # More outputs than elements each reduces: the tiled kernel, unless tiles of 32 outputs
    # launch enough blocks, and one thread per output where a row of outputs fills the device
    for (m, n, A, shape) in ((4096, 1024, Int32, :tiled_strided), (8192, 300, Int32, :columns),
                             (8192, 300, Int64, :by_thread))
        h = rand(A(-9):A(9), m, n)
        a = array_from_host(h)
        TEST_KERNELS && @test shape_of(a, (2,), A) === shape
        r = AK.reduce(+, a; dims=2, init=zero(A), alg)
        @test eltype(r) === A && Array(r) == sum(h; dims=2)
    end
    # Runs of contiguous outputs at the offsets of other kept dimensions: the middle dimension of
    # 3-d and 4-d arrays, and a view at an offset whose kept dimensions are not adjacent in memory
    h = rand(Int32(-9):Int32(9), 50, 30_000, 7)
    for (a, ha) in ((array_from_host(rand(Int32(-9):Int32(9), 96, 7000, 10)), nothing),
                    (array_from_host(rand(Int32(-9):Int32(9), 40, 30_000, 3, 2)), nothing),
                    (view(array_from_host(h), 3:42, :, 1:2:7), h[3:42, :, 1:2:7]))
        ha = something(ha, Array(a))
        TEST_KERNELS && @test shape_of(a, (2,), Int32) === :columns
        @test Array(AK.reduce(+, a; dims=2, init=Int32(0), alg)) == sum(ha; dims=2)
    end
end


@testset "reductions into a single output" begin
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
        # ... in the accumulator type, and without a neutral element
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
        # ... with a workspace, used twice
        R = array_from_host(Int8[0])
        ws = AK.workspace(AK.mapreducedim!, identity, +, R, v8; acctype=Int32, alg)
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, alg, workspace=ws)
        @test Array(R) == Int8[5]
        AK.mapreducedim!(identity, +, R, v8; acctype=Int32, alg, workspace=ws)
        @test Array(R) == Int8[10]
    end
end


@testset "lazy sources" begin
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
        # (without a neutral element)
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


@testset "offset axes" begin
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
        # ... and one with explicit offset axes, larger than its argument's
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


@testset "reduction contract" begin
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
    # (`init` is not a partial result: only the type of `op(init, x)` counts)
    @test AK.reduce((a, b) -> something(a, 0) + something(b, 0), array_from_host(Int32[1, 2]);
                    init=nothing, alg) === 3
    @test AK.reduce((a, b) -> Base.add_sum(a, b), v8; alg) === 227
    @test AK.maximum(v8; alg) === Int8(100)
    # ... joined with the one-element partial results': here the element type, not the narrower
    # one the fold produces
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
    # ... which `init` does not change (only an empty reduction returns it as it is)
    @test AK.sum(array_from_host(Int32[1, 2]); init=0, acctype=Int32, alg) === Int32(3)
    @test AK.count(array_from_host([true, true]); acctype=Int32, alg) === Int32(2)
    @test AK.sum(array_from_host(Int32[]); init=0, acctype=Int32, alg) === 0
    if !TEST_KERNELS
        @test_throws InexactError AK.reduce(+, Int8[1, 2]; init=0.5, acctype=Int16, alg)
    end
    # ... and must be able to hold the partial results
    @test_throws ArgumentError AK.sum(v8; acctype=String, alg)
    @test_throws ArgumentError AK.reduce((a, b) -> "", v8; acctype=Int, alg)
    @test_throws ArgumentError AK.sum(v8; acctype=1, alg)
    # (whether their values fit is the caller's obligation)
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
    # ... an error only where an output has an empty slice
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

@testset "non-commutative reductions" begin
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
    # ... finishing on the host, from the source or from the first pass's partial results
    xh = [randmat2() for _ in 1:20_000]
    x = array_from_host(xh)
    for switch_below in (30_000, 100)
        alg = reduce_alg(; block_size=32, items_per_thread=2, switch_below)
        @test AK.reduce(matmul2, x; commutative=false, alg) == foldl(matmul2, xh)
    end
    # ... sources read with vector loads, contiguous at an offset, strided, fused
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
    # ... and a workspace, used twice
    ws = AK.workspace(AK.reduce, matmul2, x; commutative=false, alg=REDUCE_ALG)
    for _ in 1:2
        @test AK.reduce(matmul2, x; commutative=false, alg=REDUCE_ALG, workspace=ws) ==
              foldl(matmul2, xh)
    end

    # Along `dims`, the launch shape of every commutative one in order: one thread per output
    # (for those taking one, and short runs of consecutive elements), the columns kernel (in one
    # pass or two, for columns and tiles of strided slices), and a block per output, or several
    # (for long runs, few outputs, and the other shapes); also non-adjacent dimensions. With and
    # without a neutral element.
    shapes = Set{Pair{Symbol, Symbol}}()
    cases = (((3, 2000), 1), ((2000, 3), 1), ((2000, 3), 2), ((40, 5, 30), 1), ((40, 5, 30), 2),
             ((40, 5, 30), 3), ((40, 5, 30), (1, 3)), ((9, 4, 3000), (2, 3)), ((8, 300_000), 2),
             ((40, 70_000), 2), ((256, 256), 2), ((300, 300), 2), ((3000, 300), 1),
             ((300_000, 2), 1), ((70_000, 40), 1), ((64, 4000), 1), ((512, 40), 1),
             ((1, 5000, 2), 2), ((40, 3000, 30), 2))
    for (sz, dims) in cases
        if TEST_KERNELS
            segs = AK.merge_dims(sz, AK.dense_strides(sz), AK.reduced_dims(dims, length(sz)))
            dst_size = prod(d in dims ? 1 : sz[d] for d in eachindex(sz))
            len = prod(sz) ÷ dst_size
            block_size = AK.resolve_reduce(REDUCE_ALG, BACKEND, Mat2, dims).block_size
            shape, groups = AK.reduce_shape(dst_size, len, segs, block_size,
                                            AK.reduce_tuning(BACKEND, Mat2), Mat2)
            push!(shapes,
                  shape => AK.ordered_shape(shape, groups, dst_size, len, block_size)[1])
        end
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
    if TEST_KERNELS
        @test Set(last.(shapes)) == Set((:by_thread, :columns_ordered, :ordered))
        # (with the default settings, every shape is taken; tunings can rule some out)
        t = AK.reduce_tuning(BACKEND, Mat2)
        if AK.resolve_reduce(REDUCE_ALG, BACKEND, Mat2, 1).block_size == 256 &&
           t.columns_min_elements <= 1 << 18
            required = Set((:by_thread => :by_thread,
                            :columns => :columns_ordered, :tiled_strided => :columns_ordered,
                            :by_block => :ordered, :by_block_each => :ordered,
                            :multigroup => :ordered))
            t.segments_short_runs && push!(required, :by_lanes => :by_thread)
            # (the (512, 40) case has 256 bytes per lane)
            t.segments_lane_bytes >= 256 && push!(required, :by_lanes => :ordered)
            @test shapes ⊇ required
        end
    end
    # ... square tiles in blocks too small for the columns kernel
    xh = [randmat2() for _ in 1:256, _ in 1:256]
    @test Array(AK.reduce(matmul2, array_from_host(xh); dims=2, commutative=false,
                          alg=reduce_alg(block_size=8))) ==
          reduce(matmul2, xh; dims=2, init=I2)
    # ... sources read with vector loads (contiguous, at an offset), strided views and lazy ones;
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
