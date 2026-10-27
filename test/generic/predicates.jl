# The algorithms under test: the kernel algorithms the backend supports, or the threaded host
# algorithm, each with default and explicit settings; and `Auto()`.
PRED_ALGS = if TEST_KERNELS
    AK.Algorithm[
        AK.ConcurrentWrite(), AK.ConcurrentWrite(block_size=64),
        AK.ViaReduce(), AK.ViaReduce(AK.BlockReduce(block_size=64, switch_below=100))]
else
    AK.Algorithm[AK.CPUThreads.Partitioned(), AK.CPUThreads.Partitioned(max_tasks=2, min_elems=100)]
end
HOST_KERNELS || pushfirst!(PRED_ALGS, AK.Auto())

@testset "any and all: $alg" for alg in PRED_ALGS

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
    # Offset axes
    @test AK.any(iszero, Base.IdentityUnitRange(0:999); backend=BACKEND, alg) === true
    @test AK.all(i -> i < 1000, Base.IdentityUnitRange(0:999); backend=BACKEND, alg) === true
    v = offset_from_host(Int32.(1:1000), (-5,))
    @test AK.any(==(1000), v; alg) === true
    @test AK.all(<(1000), v; alg) === false
    v = offset_from_host(Int32[5], (-5,))
    @test AK.any(==(5), v; alg) === true
    @test AK.all(==(5), v; alg) === true

    # The predicate must return a Bool, as in Base
    @test_throws ArgumentError AK.any(identity, array_from_host(Int32[2, 3]); alg)
    @test_throws ArgumentError AK.all(identity, array_from_host(Int32[2, 3]); alg)
    # It is never called on an empty array
    @test AK.any(identity, array_from_host(Int32[]); alg) === false
    @test AK.all(identity, array_from_host(Int32[]); alg) === true

    # A reshaped view, whose wrappers `@Const` rebuilds on the device
    hr = rand(Int32(0):Int32(9), 50, 40)
    vr = vec(view(array_from_host(hr), 1:40, 1:30))
    @test AK.any(x -> x > 8, vr; alg) === any(x -> x > 8, vec(view(hr, 1:40, 1:30)))
    @test AK.all(x -> x < 10, vr; alg) === true

    # A source of a bits-union element type (where the backend's arrays can hold one)
    if TEST_BITS_UNIONS
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

# Julia does not specialize a method on a function argument that it only passes on. A call then
# runs code compiled for any `Function`, which for `any`, `all` and `findall` inferred the
# predicate's result type at run time, on every call. The methods that take the predicate name
# its type; after calls of each form, none of their specializations may have been widened to
# `Function`.
@testset "predicates: specialized on the predicate" begin
    v = array_from_host(rand(Float32, 1000))
    for alg in PRED_ALGS
        AK.any(isnan, v; alg); AK.all(x -> x < 2f0, v; alg)
        with_workspace(AK.any, isnan, v; alg); with_workspace(AK.all, x -> x < 2f0, v; alg)
    end
    AK.count(isnan, v; alg=REDUCE_ALG); with_workspace(AK.count, isnan, v; alg=REDUCE_ALG)
    AK.findall(isnan, v; alg=FINDALL_ALG); with_workspace(AK.findall, isnan, v; alg=FINDALL_ALG)

    # The method instances that name one of the operations (their methods, keyword bodies and
    # sorters, and plans), and those of the internal methods that take the predicate
    ops = (AK.any, AK.all, AK.count, AK.findall)
    internal = (AK.check_bool_result, AK.predicate_setup, AK.any_run, AK.all_run,
                AK.findall_run)
    fs = Any[Core.kwcall]
    for name in names(AK; all=true)
        isdefined(AK, name) && getfield(AK, name) isa Function && push!(fs, getfield(AK, name))
    end
    widened = String[]
    for f in unique(fs), m in methods(f)
        parentmodule(m) === AK || continue
        for mi in Base.specializations(m)
            params = Base.unwrap_unionall(mi.specTypes).parameters
            f in internal || Base.any(op -> typeof(op) in params, ops) || continue
            Function in params && push!(widened, string(mi.specTypes))
        end
    end
    @test isempty(widened)
end
