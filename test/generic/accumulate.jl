# The whole-array scan algorithms under test: each kernel algorithm the backend supports, or the
# threaded host algorithm. `scan_alg` and `slice_alg` build a whole-array or `dims` algorithm for
# the configuration from both kinds of settings; without settings they give `Auto()`, except in
# the `--cpu-ka` configuration, whose point is to run AK's kernels on the host backend.
SCAN_ALGS = if !TEST_KERNELS
    [AK.CPUThreads.Partitioned]
elseif TEST_DL
    [AK.ScanPrefixes, AK.DecoupledLookback]
else
    [AK.ScanPrefixes]
end
scan_name(A) = A === AK.CPUThreads.Partitioned ? "threads" : string(nameof(A))

function scan_alg(A=nothing; block_size=nothing, items_per_thread=nothing,
                  max_tasks=nothing, min_elems=nothing)
    if A === nothing
        Base.all(isnothing, (block_size, items_per_thread, max_tasks, min_elems)) &&
            return HOST_KERNELS ? AK.ScanPrefixes() : AK.Auto()
        A = TEST_KERNELS ? AK.ScanPrefixes : AK.CPUThreads.Partitioned
    end
    A === AK.CPUThreads.Partitioned ? A(; max_tasks, min_elems) : A(; block_size, items_per_thread)
end

function slice_alg(; block_size=nothing, max_tasks=nothing, min_elems=nothing)
    Base.all(isnothing, (block_size, max_tasks, min_elems)) &&
        return HOST_KERNELS ? AK.SliceScan() : AK.Auto()
    TEST_KERNELS ? AK.SliceScan(; block_size) : AK.CPUThreads.Partitioned(; max_tasks, min_elems)
end

@testset "accumulate_1d $(scan_name(A))" for A in SCAN_ALGS

    Random.seed!(0)

    # Single-block exclusive scan
    for num_elems in 1:256
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; init=0, inclusive=false, alg=scan_alg(A; block_size=128))
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Single block inclusive scan
    for num_elems in 1:256
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; init=0, alg=scan_alg(A; block_size=128))
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Large exclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(ones(Int32, num_elems))
        y = copy(x)
        AK.accumulate!(+, y; init=0, inclusive=false, alg=scan_alg(A))
        yh = Array(y)
        @test all(yh .== 0:length(yh) - 1)
    end

    # Non-uniform data exposes block-carry bugs that all-ones data masks.
    for items_per_thread in (1, 3, 8)
        for _ in 1:50
            num_elems = rand(513:100_000)
            block_size = rand([16, 32, 64, 128, 256])
            init = rand(Int32(-100):Int32(100))
            xh = rand(Int32(-9):Int32(9), num_elems)
            y = array_from_host(xh)
            AK.accumulate!(+, y; init, inclusive=false,
                           alg=scan_alg(A; block_size, items_per_thread))
            @test Array(y) == (cumsum(xh) .- xh) .+ init
        end
    end

    # The default limits shared-memory use for wide element types.
    if KernelAbstractions.supports_float64(BACKEND)
        xh = ComplexF64.(1:4097)
        y = array_from_host(xh)
        AK.accumulate!(+, y; init=0.0 + 0.0im, alg=scan_alg(A))
        @test Array(y) == cumsum(xh)
    end

    # Large inclusive scan
    for _ in 1:1000
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; init=0, alg=scan_alg(A))
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Stress-testing small block sizes -> many blocks
    for _ in 1:100
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = copy(x)
        AK.accumulate!(+, y; init=0, alg=scan_alg(A; block_size=16))
        @test all(Array(y) .== accumulate(+, Array(x)))
    end

    # Allowing N-dimensional arrays, still reduced as 1D
    for _ in 1:100
        n1 = rand(1:100)
        n2 = rand(1:100)
        n3 = rand(1:100)
        vh = rand(Float32, n1, n2, n3)
        v = array_from_host(vh)
        AK.accumulate!(+, v; init=0, alg=scan_alg(A))
        @test all(Array(v) .≈ accumulate(+, vh))
    end

    # Ensuring the init value is respected
    for _ in 1:100
        num_elems = rand(1:100_000)
        x = array_from_host(rand(1:1000, num_elems), Int32)
        y = similar(x)
        init = rand(-1000:1000)
        AK.accumulate!(+, y, x; init=Int32(init), alg=scan_alg(A))
        @test all(Array(y) .== accumulate(+, Array(x); init))
    end

    # Exclusive scan
    x = array_from_host(ones(Int32, 10))
    y = copy(x)
    AK.accumulate!(+, y; init=0, inclusive=false, alg=scan_alg(A))
    @test all(Array(y) .== 0:9)

    # Test init value is respected with exclusive scan too
    x = array_from_host(ones(Int32, 10))
    y = copy(x)
    init = 10
    AK.accumulate!(+, y; init=Int32(init), inclusive=false, alg=scan_alg(A))
    @test all(Array(y) .== 10:19)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.accumulate(+, y; init=10, dims=2, inclusive=false, bad=:kwarg,
                                           alg=slice_alg())

    # Oversized temporaries are allowed.
    y = array_from_host(ones(Int32, 1000))
    AK.accumulate!(+, y; init=0, inclusive=false, temp=array_from_host(zeros(Int32, 1000)),
                   temp_flags=array_from_host(zeros(Int8, 1000)), alg=scan_alg(A; block_size=128))
    @test Array(y) == 0:999

    y = AK.accumulate(+, array_from_host(ones(Int32, 1000)); init=0, inclusive=false,
                      temp=array_from_host(zeros(Int64, 1000)),
                      temp_flags=array_from_host(zeros(Int8, 1000)),
                      alg=scan_alg(A; block_size=128))
    @test Array(y) == 0:999

    # Cross-block coherence: small tiles (block_size 16-64, 1 item/thread) maximise the number of
    # inter-block publish/consume handoffs. For DecoupledLookback each handoff relies on the
    # device-scope fence, so many-block non-uniform scans in both directions guard against a fence
    # that is not device scoped (the incoherent lookback would drop whole-block carries).
    for _ in 1:100
        num_elems = rand(5_000:200_000)
        block_size = rand((16, 32, 64))
        xh = rand(Int32(-9):Int32(9), num_elems)

        yi = array_from_host(xh)
        AK.accumulate!(+, yi; init=Int32(0), inclusive=true,
                       alg=scan_alg(A; block_size, items_per_thread=1))
        @test Array(yi) == cumsum(xh)

        init = rand(Int32(-50):Int32(50))
        ye = array_from_host(xh)
        AK.accumulate!(+, ye; init, inclusive=false,
                       alg=scan_alg(A; block_size, items_per_thread=1))
        @test Array(ye) == (cumsum(xh) .- xh) .+ init
    end
end


# An associative but non-commutative operator: 2x2 matrix products over UInt32, which wraps and so
# forms a ring. Random products of SL(2, Z) generators stay invertible, so running products never
# collapse to zero (as products of arbitrary matrices do, hiding ordering mistakes on long inputs).
struct ScanMat2
    a::UInt32
    b::UInt32
    c::UInt32
    d::UInt32
end
Base.zero(::Type{ScanMat2}) = ScanMat2(0, 0, 0, 0)
scan_matmul(x::ScanMat2, y::ScanMat2) = ScanMat2(
    x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d,
    x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d,
)
const SCAN_I2 = ScanMat2(1, 0, 0, 1)
const SCAN_GENS = (ScanMat2(1, 1, 0, 1), ScanMat2(1, 0, 1, 1),
                   ScanMat2(1, typemax(UInt32), 0, 1), ScanMat2(1, 0, typemax(UInt32), 1))
scan_randmat() = foldl(scan_matmul, rand(SCAN_GENS, 3); init=SCAN_I2)

# Sequential reference scan of each slice along `dims` (a vector is one slice)
function scan_reference(xh, dims; init, inclusive)
    out = similar(xh)
    for I in CartesianIndices(Base.setindex(axes(xh), 1:1, dims))
        acc = init
        for k in axes(xh, dims)
            J = Base.setindex(Tuple(I), k, dims)
            if inclusive
                acc = scan_matmul(acc, xh[J...])
                out[J...] = acc
            else
                out[J...] = acc
                acc = scan_matmul(acc, xh[J...])
            end
        end
    end
    out
end


@testset "accumulate_1d non-commutative $(scan_name(A))" for A in SCAN_ALGS
    Random.seed!(0)

    # Single and multiple blocks, and more blocks than one block can scan (block_size=16), so that
    # the block scan, the lookback / block-prefix carry and the chunked prefix carry are all
    # exercised. On the CPU, `max_tasks=4` exercises the carry between tasks.
    for inclusive in (true, false), (block_size, items_per_thread) in ((256, nothing), (16, 1), (32, 3))
        # Includes exact tile boundaries (16 and 96 elements) and one past them
        for n in (1, 2, 5, 16, 17, 96, 97, 100, 1000, 5000, 70_000)
            xh = [scan_randmat() for _ in 1:n]
            init = scan_randmat()
            y = array_from_host(xh)
            AK.accumulate!(scan_matmul, y; init, neutral=SCAN_I2, inclusive,
                           alg=scan_alg(A; max_tasks=4, block_size, items_per_thread))
            @test Array(y) == scan_reference(xh, 1; init, inclusive)
        end
    end
end


@testset "accumulate_nd non-commutative" begin
    Random.seed!(0)

    # Both GPU strategies: one thread per slice when there are more slices than elements per
    # slice, else one block per slice, processing the slice in several chunks of 2 * block_size
    for inclusive in (true, false), block_size in (64, 256)
        # Slices of exactly one and two chunks, and one element more, for both block sizes
        chunk = 2 * block_size
        for (sz, dims) in (((3, 2000), 2), ((2000, 3), 1), ((2000, 3), 2), ((7, 600), 2),
                           ((40, 5, 30), 1), ((40, 5, 30), 2), ((40, 5, 30), 3),
                           ((3, chunk), 2), ((3, chunk + 1), 2), ((3, 2chunk), 2), ((2chunk + 1, 3), 1))
            xh = [scan_randmat() for _ in CartesianIndices(sz)]
            init = scan_randmat()
            y = array_from_host(xh)
            AK.accumulate!(scan_matmul, y; init, neutral=SCAN_I2, inclusive, dims,
                           alg=slice_alg(; max_tasks=4, block_size))
            @test Array(y) == scan_reference(xh, dims; init, inclusive)
        end
    end
end


@testset "accumulate_nd" begin
    Random.seed!(0)

    # Test all possible corner cases against Base.accumulate
    for dims in 1:4
        for isize in 0:3
            for jsize in 0:3
                for ksize in 0:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.accumulate(+, s; init=Int32(0), dims, alg=slice_alg())

                    dh = Array(d)
                    dhres = accumulate(+, sh; init=Int32(0), dims)
                    @test dh == dhres
                    @test eltype(dh) == eltype(dhres)
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

            s = AK.accumulate(+, v; init=Int32(0), dims, alg=slice_alg())
            sh = Array(s)
            @test sh == accumulate(+, vh; init=Int32(0), dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)

            s = AK.accumulate(+, v; init=UInt32(0), dims, alg=slice_alg())
            sh = Array(s)
            @test sh == accumulate(+, vh; init=UInt32(0), dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)

            s = AK.accumulate(+, v; init=Float32(0), dims, alg=slice_alg())
            sh = Array(s)
            @test all(sh .≈ accumulate(+, vh; init=Float32(0), dims))
        end
    end

    # Ensure the init value is respected
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            init = rand(-1000:1000)
            s = AK.accumulate(+, v; init=Float32(init), dims, alg=slice_alg())
            sh = Array(s)
            @test all(sh .≈ accumulate(+, vh; init=Float32(init), dims))
        end
    end

    # Exclusive scan
    vh = ones(Int32, 10, 10)
    v = array_from_host(vh)
    s = AK.accumulate(+, v; init=0, dims=2, inclusive=false, alg=slice_alg())
    sh = Array(s)
    @test all([sh[i, :] == 0:9 for i in 1:10])

    # Test init value is respected with exclusive scan too
    vh = ones(Int32, 10, 10)
    v = array_from_host(vh)
    s = AK.accumulate(+, v; init=10, dims=2, inclusive=false, alg=slice_alg())
    sh = Array(s)
    @test all([sh[i, :] == 10:19 for i in 1:10])

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.accumulate(+, v; init=10, dims=2, inclusive=false, bad=:kwarg,
                                           alg=slice_alg())

    # Test all options with bigger matrices
    for D in [(1_000_000,3), (3,1_000_000)], dims in [1,2]
        @testset let D = D, dims = dims
            vh = ones(Float32, D)
            v = array_from_host(vh)
            s = AK.accumulate(+, v; init=0, dims)
            sh = Array(s)
            @test sh == accumulate(+, vh; init=0, dims)
        end
    end

    # Testing different settings
    AK.accumulate((x, y) -> x + 1, array_from_host(rand(Int32, 3, 4, 5)); init=Int32(0),
                  neutral=Int32(0), dims=2, alg=slice_alg(; block_size=64))
    AK.accumulate((x, y) -> x + 1, array_from_host(rand(Int32, 3, 4, 5)); init=Int32(0),
                  neutral=Int32(0), dims=3, alg=slice_alg(; block_size=64))
    # The temporaries only apply to whole-array scans
    @test_throws ArgumentError AK.accumulate(+, array_from_host(rand(Int32, 3, 4)); init=Int32(0),
                                             dims=2, temp=array_from_host(zeros(Int32, 3)))
end
@testset "cumsum" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    vh = Array(v)
    @test Array(AK.cumsum(v; alg=scan_alg())) == cumsum(vh)

    # Fuzzy testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        vh = rand(Float32, num_elems)
        v = array_from_host(vh)
        @test all(Array(AK.cumsum(v; alg=scan_alg())) .≈ cumsum(vh))
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear; not supported in Base
            # @test all(Array(AK.cumsum(v; alg=scan_alg())) .== cumsum(vh))

            # Along dimensions
            r = Array(AK.cumsum(v; dims, alg=slice_alg()))
            rh = cumsum(vh; dims)

            @test r == rh
        end
    end

    # Test promotion to op-dictated type
    xh = rand(Bool, 16)
    x = array_from_host(xh)
    @test Array(AK.cumsum(x; alg=scan_alg())) == cumsum(xh)

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.cumsum(v; alg=scan_alg(; block_size=64))

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.cumsum(v; init=10, bad=:kwarg, alg=scan_alg())

    # The other settings are stress-tested in reduce
end


@testset "cumprod" begin

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)
    vh = Array(v)
    @test Array(AK.cumprod(v; alg=scan_alg())) == cumprod(vh)

    vh = ones(Float32, 100_000)
    v = array_from_host(vh)
    @test Array(AK.cumprod(v; alg=scan_alg())) == vh

    # Fuzzy testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:10)
            n2 = rand(1:10)
            n3 = rand(1:10)
            vh = rand(Int32(-5):Int32(5), n1, n2, n3)
            v = array_from_host(vh)

            # Indexing into array as if linear; not supported in Base
            # @test all(Array(AK.cumprod(v; alg=scan_alg())) .== cumprod(vh))

            # Along dimensions
            r = Array(AK.cumprod(v; dims, alg=slice_alg()))
            rh = cumprod(vh; dims)

            @test r == rh
        end
    end

    # Testing different settings
    v = array_from_host(rand(-5:5, 100_000))
    AK.cumprod(v; alg=scan_alg(; block_size=64))

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.cumprod(v; init=10, bad=:kwarg, alg=scan_alg())

    # The other settings are stress-tested in reduce
end
