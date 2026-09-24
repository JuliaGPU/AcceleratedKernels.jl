# The algorithms under test: the kernel algorithms the backend supports, or the threaded host
# algorithm, each with default and explicit settings; and `Auto()`.
PRED_ALGS = if TEST_KERNELS
    AK.Algorithm[
        (AK._supports_concurrent_write(BACKEND) ?
            [AK.ConcurrentWrite(), AK.ConcurrentWrite(block_size=64)] : [])...,
        AK.ViaReduce(), AK.ViaReduce(AK.BlockReduce(block_size=64, switch_below=100))]
else
    AK.Algorithm[AK.CPUThreads.Partitioned(), AK.CPUThreads.Partitioned(max_tasks=2, min_elems=100)]
end
HOST_KERNELS || pushfirst!(PRED_ALGS, AK.Auto())

@testset "truth $alg" for alg in PRED_ALGS

    Random.seed!(0)

    # Simple correctness tests
    v = array_from_host(1:100)

    @test AK.any(x->x<0, v; alg) === false
    @test AK.any(x->x>99, v; alg) === true

    @test AK.all(x->x>0, v; alg) === true
    @test AK.all(x->x<100, v; alg) === false

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        @test AK.any(x->x<0, v; alg) === false
        @test AK.any(x->x<1, v; alg) === true
        @test AK.all(x->x<1, v; alg) === true
        @test AK.all(x->x<0, v; alg) === false
    end

    # Empty inputs, as Base
    e = array_from_host(Float32[])
    @test AK.any(x->x<0, e; alg) === false
    @test AK.all(x->x<0, e; alg) === true

    # Unmaterialised index ranges, which need `backend`
    x = array_from_host(rand(Float32, 1000))
    @test AK.any(i -> x[i] > 2, 1:length(x); backend=BACKEND, alg) === false
    @test AK.all(i -> x[i] < 2, 1:length(x); backend=BACKEND, alg) === true

    # The predicate must return a Bool, as in Base
    @test_throws ArgumentError AK.any(identity, array_from_host(Int32[2, 3]); alg)
    @test_throws ArgumentError AK.all(identity, array_from_host(Int32[2, 3]); alg)
    # ... which it never calls on an empty array
    @test AK.any(identity, array_from_host(Int32[]); alg) === false
    @test AK.all(identity, array_from_host(Int32[]); alg) === true

    # A reshaped view, whose wrappers `@Const` cannot rebuild on the device
    hr = rand(Int32(0):Int32(9), 50, 40)
    vr = vec(view(array_from_host(hr), 1:40, 1:30))
    @test AK.any(x -> x > 8, vr; alg) === any(x -> x > 8, vec(view(hr, 1:40, 1:30)))
    @test AK.all(x -> x < 10, vr; alg) === true

    # A source of a bits-union element type (where the backend's arrays can hold one)
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
    if unions
        hm = rand([true, false, missing], 10_000)
        dm = array_from_host(hm)
        @test AK.any(ismissing, dm; alg) === true
        @test AK.all(x -> x !== true, dm; alg) === false
        @test AK.any(x -> x === true, array_from_host(Union{Missing, Bool}[missing, false]); alg) ===
              false
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.any(x->x<0, v; alg, bad=:kwarg)
end


# GPU backends where concurrent writes are unsafe (like oneAPI) and safe
struct PredicateResolveTestBackend <: KernelAbstractions.GPU end
AK._supports_concurrent_write(::PredicateResolveTestBackend) = false
struct PredicateCWTestBackend <: KernelAbstractions.GPU end

@testset "predicate resolution" begin
    B = PredicateResolveTestBackend()
    host = AK.HOST_BACKEND
    @test AK._resolve_predicate(AK.Auto(), host, Float32) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    @test AK._resolve_predicate(AK.Auto(), PredicateCWTestBackend(), Float32) ===
          AK.ConcurrentWrite(256)
    @test AK._resolve_predicate(AK.Auto(), B, Float32) === AK.ViaReduce(AK.BlockReduce(256, 2, 0))
    # The nested reduction is resolved like any reduction
    @test AK._resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=64)), B, Float32) ===
          AK.ViaReduce(AK.BlockReduce(64, 2, 0))
    @test_throws ArgumentError AK._resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=3)), B, Float32)
    # ConcurrentWrite is rejected where it is unsafe, whatever the caller asks for
    @test_throws ArgumentError AK._resolve_predicate(AK.ConcurrentWrite(), B, Float32)
    for bad in (AK.ConcurrentWrite(block_size=0), AK.ConcurrentWrite(block_size=96),
                AK.ConcurrentWrite(block_size=2048))
        @test_throws ArgumentError AK._resolve_predicate(bad, PredicateCWTestBackend(), Float32)
    end
    # Capabilities and other families
    @test_throws ArgumentError AK._resolve_predicate(AK.CPUThreads.Partitioned(), B, Float32)
    @test_throws ArgumentError AK._resolve_predicate(AK.BlockReduce(), B, Float32)
    @test_throws ArgumentError AK._resolve_predicate(AK.ScanScatter(), B, Float32)
end
