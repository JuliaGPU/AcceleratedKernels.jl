# Resolution of reduction algorithms, without launching kernels: resolution only looks at types
# and `dims`, so host arrays stand in for device arrays where an entry point is called.

# A GPU backend with a tuning the tests can change
struct ReduceResolveTestBackend <: KernelAbstractions.GPU end
const REDUCE_RESOLVE_TUNING = Ref(AK.ReduceTuning())
AK.reduce_tuning(::ReduceResolveTestBackend, ::Type) = REDUCE_RESOLVE_TUNING[]

# A backend that cannot run AK's kernels, like KernelAbstractions 0.9's `CPU`
struct ReduceNoKernelsTestBackend <: KernelAbstractions.GPU end
AK._runs_kernels(::ReduceNoKernelsTestBackend) = false

const RRB = ReduceResolveTestBackend()

resolve_reduce(alg; T=Float32, dims=:, backend=RRB) = AK._resolve_reduce(alg, backend, T, dims)

function with_reduce_tuning(f; kwargs...)
    old = REDUCE_RESOLVE_TUNING[]
    REDUCE_RESOLVE_TUNING[] = AK.ReduceTuning(; kwargs...)
    try
        f()
    finally
        REDUCE_RESOLVE_TUNING[] = old
    end
end


@testset "reduce resolution: Auto" begin
    # The default tuning reproduces AK's historical settings, for whole arrays and `dims`
    @test resolve_reduce(AK.Auto()) === AK.BlockReduce(256, 2, 0)
    @test resolve_reduce(AK.Auto(); dims=nothing) === AK.BlockReduce(256, 2, 0)
    @test resolve_reduce(AK.Auto(); dims=2) === AK.BlockReduce(256, 2, 0)
    @test resolve_reduce(AK.Auto(); dims=(1, 3)) === AK.BlockReduce(256, 2, 0)

    # On the host backend: the threaded algorithm, filled
    host = AK.HOST_BACKEND
    @test resolve_reduce(AK.Auto(); backend=host) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    @test resolve_reduce(AK.Auto(); backend=host, dims=1) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)

    # The tuning fills every field; `stable` means nothing to a reduction
    with_reduce_tuning(; block_size=512, items_per_thread=4, switch_below=100) do
        @test resolve_reduce(AK.Auto()) === AK.BlockReduce(512, 4, 100)
        @test resolve_reduce(AK.Auto(stable=false)) === AK.BlockReduce(512, 4, 100)
    end
end


@testset "reduce resolution: explicit algorithms" begin
    # Explicit fields win over the tuning, unset ones come from it
    with_reduce_tuning(; block_size=512, items_per_thread=4, switch_below=100,
                         threads_min_elems=7) do
        @test resolve_reduce(AK.BlockReduce()) === AK.BlockReduce(512, 4, 100)
        @test resolve_reduce(AK.BlockReduce(block_size=64)) === AK.BlockReduce(64, 4, 100)
        @test resolve_reduce(AK.BlockReduce(items_per_thread=1, switch_below=0)) ===
              AK.BlockReduce(512, 1, 0)
        @test resolve_reduce(AK.BlockReduce(block_size=128); dims=1) === AK.BlockReduce(128, 4, 100)
    end
    host = AK.HOST_BACKEND
    @test resolve_reduce(AK.CPUThreads.Partitioned(max_tasks=3); backend=host) ===
          AK.CPUThreads.Partitioned(3, 1)
    @test resolve_reduce(AK.CPUThreads.Partitioned(min_elems=1000); backend=host) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1000)

    # Domains of explicit fields
    for bad in (AK.BlockReduce(block_size=0), AK.BlockReduce(block_size=192),
                AK.BlockReduce(block_size=2048), AK.BlockReduce(items_per_thread=0),
                AK.BlockReduce(switch_below=-1))
        @test_throws ArgumentError resolve_reduce(bad)
    end
    for bad in (AK.CPUThreads.Partitioned(max_tasks=0), AK.CPUThreads.Partitioned(min_elems=0))
        @test_throws ArgumentError resolve_reduce(bad; backend=host)
    end

    # `items_per_thread` and `switch_below` only apply to whole-array reductions; they used to be
    # ignored silently along `dims`
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=1)
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(switch_below=0); dims=(1, 2))
    @test resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=nothing) isa AK.BlockReduce

    # Whole-array reductions need tiles of at least two elements (one never shrinks the input)
    # and at most typemax(Int32), whether the settings are explicit or from the tuning
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=1))
    @test resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=2)) === AK.BlockReduce(1, 2, 0)
    @test resolve_reduce(AK.BlockReduce(block_size=1); dims=1) === AK.BlockReduce(1, 2, 0)
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=1 << 56))
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1024, items_per_thread=1 << 22))
    with_reduce_tuning(; block_size=1, items_per_thread=1) do
        @test_throws ArgumentError resolve_reduce(AK.Auto())
        @test resolve_reduce(AK.Auto(); dims=2) === AK.BlockReduce(1, 1, 0)
    end

    # Capabilities: the threaded algorithm only on the host, kernels only where they run
    @test_throws ArgumentError resolve_reduce(AK.CPUThreads.Partitioned())
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); backend=ReduceNoKernelsTestBackend())
    if AK._runs_kernels(host)
        @test resolve_reduce(AK.BlockReduce(); backend=host) === AK.BlockReduce(256, 2, 0)
    else
        @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); backend=host)
    end

    # Algorithms of other families
    @test_throws ArgumentError resolve_reduce(AK.MergeSort())
    @test_throws ArgumentError resolve_reduce(AK.CPUThreads.SampleSort(); backend=host)
    @test_throws ArgumentError AK._resolve_sort(AK.BlockReduce(), RRB, zeros(Float32, 10), :,
                                                Base.Order.Forward; perm=false)
end


@testset "reduce resolution: tuning values" begin
    # A tuning cannot make a reduction along `dims` launch nothing
    with_reduce_tuning(; target_blocks=0) do
        @test_throws ArgumentError AK.mapreduce_nd!(identity, +, zeros(Int32, 1, 4),
                                                    zeros(Int32, 4, 4), RRB,
                                                    AK.BlockReduce(256, 2, 0), Int32;
                                                    init=AK._NoInit(), neutral=nothing,
                                                    dims_valid=(1,))
    end
end


@testset "reduce resolution: inference" begin
    host = typeof(AK.HOST_BACKEND)
    for (B, dims) in ((ReduceResolveTestBackend, Colon), (ReduceResolveTestBackend, Int),
                      (host, Nothing), (host, Tuple{Int, Int}))
        rt = only(Base.return_types(AK._resolve_reduce, (AK.Auto, B, Type{Float32}, dims)))
        @test rt <: Union{AK.BlockReduce, AK.CPUThreads.Partitioned}
    end
end


@testset "reduce resolution: entry points" begin
    v = rand(Int32(1):Int32(100), 1000)
    m = rand(Int32(1):Int32(100), 10, 100)

    # Invalid explicit algorithms are rejected before any data is touched, whatever the length
    for x in (v, Int32[], m)
        @test_throws ArgumentError AK.reduce(+, x; init=Int32(0), alg=AK.BlockReduce(block_size=3))
        @test_throws ArgumentError AK.mapreduce(abs, +, x; init=Int32(0), alg=AK.MergeSort())
        @test_throws ArgumentError AK.sum(x; alg=AK.CPUThreads.Partitioned(max_tasks=0))
    end
    @test_throws ArgumentError AK.reduce(+, m; init=Int32(0), dims=1,
                                         alg=AK.BlockReduce(items_per_thread=4))

    # The convenience reductions forward `alg`
    alg = AK.CPUThreads.Partitioned(max_tasks=4, min_elems=10)
    @test AK.sum(v; alg) == sum(v)
    @test AK.prod(Int64.(v[1:5]); alg) == prod(Int64.(v[1:5]))
    @test AK.maximum(v; alg) == maximum(v)
    @test AK.minimum(v; alg) == minimum(v)
    @test AK.count(>(50), v; alg) == count(>(50), v)
    @test AK.sum(m; dims=2, alg) == sum(m; dims=2)

    # The loose settings are gone
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), block_size=256)
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), max_tasks=2)
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), prefer_threads=false)
    @test_throws MethodError AK.sum(v; switch_below=10)
end


@testset "reduce resolution: accumulator and seed" begin
    # The accumulator type follows Base's promotion to a fixed point
    @test AK._reduce_acctype(Base.add_sum, Int8, Int8) === Int
    @test AK._reduce_acctype(+, Int8, Int8) === Int8
    @test AK._reduce_acctype(+, Bool, Bool) === Int
    @test AK._reduce_acctype(+, Int, Float32) === Float32
    @test AK._reduce_acctype(max, Tuple{Int32, Int32}, Tuple{Int32, Int32}) === Tuple{Int32, Int32}
    @test AK._first_type(Base.add_sum, Int8) === Int
    if VERSION >= v"1.13-"      # older `add_sum`s keep mixed-signedness small integers small
        @test AK._reduce_acctype(Base.add_sum, UInt8, Int8) === Int
        @test AK._reduce_acctype(Base.add_sum, UInt8, Bool) === Int
    end
    @test AK._reduce_acctype(+, Union{}, Char) === Union{}
    @test AK._reduce_acctype(Returns(false), Union{}, Int) === Int
    @test AK._reduce_acctype(coalesce, Int, Union{Missing, Int}) === Union{Missing, Int}
    # `init` is applied once, not a partial result: only `op(init, x)`'s type counts
    @test AK._reduce_acctype((a, b) -> something(a, 0) + something(b, 0), Nothing, Int) === Int

    # Seeds: the caller's neutral, else GPUArraysCore's, else an empty lane
    @test AK._reduce_seed(+, Int32, nothing) === Int32(0)
    @test AK._reduce_seed(min, Float32, nothing) === Inf32
    @test AK._reduce_seed(+, Float32, nothing) === -0.0f0
    @test AK._reduce_seed(Base.add_sum, ComplexF32, nothing) === complex(-0.0f0, -0.0f0)
    @test AK._reduce_seed(+, Int32, 0) === Int32(0)
    @test AK._reduce_seed((a, b) -> a + b, Int32, nothing) isa AK._Lane{Int32}
    @test !AK._valid(AK._reduce_seed((a, b) -> a + b, Int32, nothing))
    # Empty lanes of every kind of type
    @test !AK._valid(AK._Lane{Tuple{Int8, Int64}}())
    @test !AK._valid(AK._Lane{Union{Missing, Int}}())
    @test !AK._valid(AK._Lane{String}())
    @test AK._valid(AK._Lane{String}("x"))

    # Results infer, whether or not the operator has a known neutral element
    @test only(Base.return_types(v -> AK.sum(v), (Vector{Int8},))) === Int
    @test only(Base.return_types(v -> AK.reduce((a, b) -> a + b, v), (Vector{Int32},))) === Int32
    @test only(Base.return_types(m -> AK.sum(m; dims=1), (Matrix{Int8},))) === Matrix{Int}
    @test only(Base.return_types(v -> AK.mapreduce(tuple, (a, b) -> a, v, v),
                                 (Vector{Int32},))) === Tuple{Int32, Int32}

    # BlockReduce needs a bits-type accumulator; the host algorithm does not
    @test_throws ArgumentError resolve_reduce(AK.Auto(); T=Union{Missing, Int32})
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); T=String)
    @test resolve_reduce(AK.Auto(); T=String, backend=AK.HOST_BACKEND) isa AK.CPUThreads.Partitioned
end
