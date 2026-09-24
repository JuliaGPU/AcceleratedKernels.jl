# Resolution of scan algorithms, without launching kernels.

# A GPU backend with a tuning and a lookback capability the tests can change
struct ScanResolveTestBackend <: KernelAbstractions.GPU end
const SCAN_RESOLVE_TUNING = Ref(AK.ScanTuning())
const SCAN_RESOLVE_LOOKBACK = Ref(false)
AK.scan_tuning(::ScanResolveTestBackend, ::Type) = SCAN_RESOLVE_TUNING[]
AK._supports_lookback(::ScanResolveTestBackend) = SCAN_RESOLVE_LOOKBACK[]

# A backend that cannot run AK's kernels, like KernelAbstractions 0.9's `CPU`
struct ScanNoKernelsTestBackend <: KernelAbstractions.GPU end
AK._runs_kernels(::ScanNoKernelsTestBackend) = false

const SRB = ScanResolveTestBackend()

resolve_scan(alg; T=Float32, dims=nothing, backend=SRB) = AK._resolve_scan(alg, backend, T, dims)

function with_scan_tuning(f; lookback=false, kwargs...)
    old = SCAN_RESOLVE_TUNING[], SCAN_RESOLVE_LOOKBACK[]
    SCAN_RESOLVE_TUNING[] = AK.ScanTuning(; kwargs...)
    SCAN_RESOLVE_LOOKBACK[] = lookback
    try
        f()
    finally
        SCAN_RESOLVE_TUNING[], SCAN_RESOLVE_LOOKBACK[] = old
    end
end


@testset "scan resolution: Auto" begin
    # The default tuning reproduces AK's historical settings: 256 threads, at most 8 items per
    # thread, fewer for wide element types
    @test resolve_scan(AK.Auto()) === AK.ScanPrefixes(256, 8)
    @test resolve_scan(AK.Auto(); T=ComplexF64) === AK.ScanPrefixes(256, 7)
    @test resolve_scan(AK.Auto(); T=NTuple{16, Float64}) === AK.ScanPrefixes(256, 1)
    @test resolve_scan(AK.Auto(); dims=2) === AK.SliceScan(256)

    # DecoupledLookback only where the tuning prefers it and the backend supports it
    with_scan_tuning(; prefer_lookback=true) do
        @test resolve_scan(AK.Auto()) === AK.ScanPrefixes(256, 8)
    end
    with_scan_tuning(; lookback=true) do
        @test resolve_scan(AK.Auto()) === AK.ScanPrefixes(256, 8)
    end
    with_scan_tuning(; prefer_lookback=true, lookback=true) do
        @test resolve_scan(AK.Auto()) === AK.DecoupledLookback(256, 8)
        @test resolve_scan(AK.Auto(); dims=1) === AK.SliceScan(256)
    end

    # On the host backend: the threaded algorithm, filled
    host = AK.HOST_BACKEND
    @test resolve_scan(AK.Auto(); backend=host) === AK.CPUThreads.Partitioned(Threads.nthreads(), 2)
    @test resolve_scan(AK.Auto(); backend=host, dims=1) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 2)
end


@testset "scan resolution: explicit algorithms" begin
    # Explicit fields win over the tuning; `items_per_thread` is derived from the effective
    # `block_size`, so an explicit block size gets a matching default
    with_scan_tuning(; block_size=128, max_items=16) do
        @test resolve_scan(AK.ScanPrefixes()) === AK.ScanPrefixes(128, 16)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024)) === AK.ScanPrefixes(1024, 7)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024, items_per_thread=12)) ===
              AK.ScanPrefixes(1024, 12)
        @test resolve_scan(AK.SliceScan(); dims=1) === AK.SliceScan(128)
    end
    with_scan_tuning(; local_mem_bytes=1024) do
        @test resolve_scan(AK.ScanPrefixes(block_size=64)) === AK.ScanPrefixes(64, 3)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024)) === AK.ScanPrefixes(1024, 1)
    end
    host = AK.HOST_BACKEND
    @test resolve_scan(AK.CPUThreads.Partitioned(max_tasks=3); backend=host) ===
          AK.CPUThreads.Partitioned(3, 2)

    # Domains of explicit fields
    for bad in (AK.ScanPrefixes(block_size=0), AK.ScanPrefixes(block_size=96),
                AK.ScanPrefixes(block_size=2048), AK.ScanPrefixes(items_per_thread=0),
                AK.SliceScan(block_size=3))
        @test_throws ArgumentError resolve_scan(bad; dims=bad isa AK.SliceScan ? 1 : nothing)
    end
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(block_size=1024, items_per_thread=1 << 22))
    # A tuning's settings are checked like explicit ones
    for block_size in (0, 1 << 62)
        with_scan_tuning(; block_size) do
            @test_throws ArgumentError resolve_scan(AK.Auto())
        end
    end
    # Exclusive scans on threads need two elements per task
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.Partitioned(min_elems=1); backend=host)

    # Whole-array algorithms do not scan along `dims`, and SliceScan only does; they used to be
    # ignored silently along `dims`
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); dims=1)
    @test_throws ArgumentError resolve_scan(AK.SliceScan())

    # DecoupledLookback needs the capability, whatever the tuning says
    @test_throws ArgumentError resolve_scan(AK.DecoupledLookback())
    with_scan_tuning(; lookback=true) do
        @test resolve_scan(AK.DecoupledLookback()) === AK.DecoupledLookback(256, 8)
    end

    # Kernels need a bits-type element
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); T=String)
    @test_throws ArgumentError resolve_scan(AK.Auto(); T=Union{Missing, Int}, dims=1)

    # Capabilities: the threaded algorithm only on the host, kernels only where they run
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.Partitioned())
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); backend=ScanNoKernelsTestBackend())

    # Algorithms of other families
    @test_throws ArgumentError resolve_scan(AK.BlockReduce())
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.SampleSort(); backend=host)
    @test_throws ArgumentError AK._resolve_reduce(AK.ScanPrefixes(), SRB, Float32, :)
end


@testset "scan resolution: inference" begin
    host = typeof(AK.HOST_BACKEND)
    for (B, dims) in ((ScanResolveTestBackend, Nothing), (ScanResolveTestBackend, Int),
                      (host, Nothing), (host, Int))
        rt = only(Base.return_types(AK._resolve_scan, (AK.Auto, B, Type{Float32}, dims)))
        @test rt <: Union{AK.ScanPrefixes, AK.DecoupledLookback, AK.SliceScan,
                          AK.CPUThreads.Partitioned}
    end
end


@testset "scan resolution: entry points" begin
    # The keyword backend: derived from the arrays, destination first
    v = rand(Int32(1):Int32(9), 1000)
    @test AK.accumulate!(+, copy(v); init=Int32(0)) == cumsum(v)
    @test AK.accumulate!(+, similar(v), v; init=Int32(0)) == cumsum(v)
    @test AK.accumulate(+, v; init=Int32(0), backend=AK.HOST_BACKEND) == cumsum(v)
    # An explicit backend is used as given, and checked against the algorithm
    @test_throws ArgumentError AK.accumulate!(+, similar(v), v; init=Int32(0), backend=SRB,
                                              alg=AK.CPUThreads.Partitioned())
    # A rejected algorithm leaves the destination untouched
    dst = zeros(Int32, 1000)
    @test_throws ArgumentError AK.accumulate!(+, dst, v; init=Int32(0),
                                              alg=AK.CPUThreads.Partitioned(min_elems=1))
    @test Base.all(iszero, dst)
    # Invalid `dims`
    @test_throws ArgumentError AK.accumulate(+, reshape(v, 10, 100); init=Int32(0), dims=0)
end
