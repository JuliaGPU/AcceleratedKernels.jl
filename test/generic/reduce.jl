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
            @test AK._contiguous_vector_width(v, 0, (1024,), (1,), 1024, 256) ==
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
        @test AK._contiguous_vector_width(r, 0, (1027,), (1,), 1024, 256) == 0
        @test Array(AK.reduce(+, @view(r[1:1024, :]); alg=REDUCE_ALG, init=0.0f0, dims=1)) ≈
            sum(@view(rh[1:1024, :]); dims=1)

        bh = rand(Int32(1):Int32(100), 260, 1024)
        b = array_from_host(bh)
        @test AK._contiguous_vector_width(b, 2, (260,), (1,), 258, 256) == 0
        @test Array(AK.reduce(+, @view(b[3:end, :]); alg=REDUCE_ALG, init=Int32(0), dims=1)) ==
            Base.reduce(+, @view(bh[3:end, :]); init=Int32(0), dims=1)
    end

    if !TEST_KERNELS
        storage = Vector{UInt8}(undef, 64)
        GC.@preserve storage begin
            offset = Int(mod(-UInt(pointer(storage)), 16)) + 4
            misaligned = unsafe_wrap(Array, Ptr{Float32}(pointer(storage) + offset), 4)
            @test AK._contiguous_vector_width(misaligned, 0, (4,), (1,), 4, 256) == 0
        end

        # The CPU fallback should not require strided storage.
        vh = reshape(1:12, 1, 3, 4)
        @test Array(AK.reduce(+, vh; alg=REDUCE_ALG, backend=BACKEND, init=0, dims=(1,2))) ==
            sum(vh; init=0, dims=(1,2))
    else
        # Strided GPU sources (views, adjoints, permuted dims) take the stride-based
        # fast path over their dense parent buffer; the offset view exercises a nonzero
        # base offset. Broadcasted/lazy sources still take the generic fallback.
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
        # base offset. Broadcasted/lazy sources still take the generic fallback.
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
            @test AK.sum(v; alg=REDUCE_ALG) == sum(vh)

            # Along dimensions
            r = Array(AK.sum(v; alg=REDUCE_ALG, dims))
            rh = sum(vh; dims)

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
    # one-block-per-output reduction (contiguous and strided sources), and the multi-block
    # reduction with its second pass.
    Random.seed!(0)
    vh = rand(Int32(1):Int32(100), 100_000)
    v = array_from_host(vh)
    shapes = (((3000, 40), 1), ((40, 3000), 2), ((20_000, 4), 1))
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
    # On the host, an abstract accumulator type with the caller's neutral element
    if !TEST_KERNELS
        @test AK.reduce(+, Real[1, 2.5, 3]; neutral=0, alg=AK.CPUThreads.Partitioned(max_tasks=2, min_elems=1)) == 6.5
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

    # A source whose wrappers `@Const` cannot rebuild on the device: a reshaped view
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
    # WORKAROUND(KernelAbstractions): `zeros`, which `array_from_host` uses, fails for these
    # element types on KernelAbstractions 0.10's POCL backend (JuliaGPU/KernelAbstractions.jl#791),
    # so `--cpu-ka` skips these tests; once fixed, only OpenCL.jl, whose arrays cannot hold them
    # yet, should skip
    unions = try
        array_from_host(Union{Missing, Bool}[missing, true])
        true
    catch
        false
    end
    unions && for (shape, dims) in (((1000, 300), 1), ((1000, 300), 2), ((256, 256), 2),
                          ((20_000, 2), (1, 2)), ((5, 3), 3))
        hm = rand([true, false, missing], shape...)
        dm = array_from_host(hm)
        @test AK.mapreduce(code, max, dm; init=0x00, alg) === mapreduce(code, max, hm)
        @test Array(AK.mapreduce(code, max, dm; dims, init=0x00, alg)) ==
              mapreduce(code, max, hm; dims)
    end
end
