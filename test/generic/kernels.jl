# Julia does not specialize a method on a function argument that it only passes on, and a GPU
# backend then specializes such a kernel again on every launch, which costs small calls a
# fraction of a microsecond each. The kernels name the types of such arguments; after running
# each family's kernels, none of their specializations may have been widened.

@testset "kernels: specialized on their arguments" begin
    if TEST_KERNELS
        Random.seed!(0)
        T = Float32
        v = array_from_host(rand(T, 10_000))
        A = array_from_host(rand(T, 300, 300))
        B = array_from_host(rand(T, 16, 64, 32))
        mymax(a, b) = max(a, b)

        # (the kernel algorithms, which `Auto()` does not pick on the host backend)
        alg = REDUCE_ALG
        AK.sum(v; alg); AK.mapreduce(abs, +, v; alg); AK.mapreduce(abs, mymax, v; init=zero(T), alg)
        AK.sum(v; commutative=false, alg)
        for d in (1, 2, (1, 2)), f in (identity, abs)
            AK.mapreduce(f, +, A; dims=d, init=zero(T), alg)
            AK.mapreduce(f, +, A; dims=d, init=zero(T), commutative=false, alg)
        end
        for d in (1, 2, 3, (1, 3))
            AK.maximum(B; dims=d, alg)
        end
        for d in (nothing, 1, 2)
            alg = d === nothing ? scan_alg() : slice_alg()
            AK.accumulate(+, A; dims=d, alg)
            AK.accumulate(mymax, A; dims=d, inclusive=false, init=zero(T), alg)
        end
        AK.accumulate(+, B; dims=2, alg=slice_alg())
        AK.findall(x -> x > 0.5f0, v; alg=FINDALL_ALG)
        for alg in (AK.ConcurrentWrite(), AK.ViaReduce(AK.BlockReduce()))
            AK.any(x -> x > 2f0, v; alg); AK.all(x -> x < 2f0, v; alg)
        end
        AK.map(sin, v); AK.reverse(v)
        for alg in (AK.MergeSort(), AK.BitonicSort())
            AK.sort(v; alg); AK.sort(A; dims=2, alg)
            AK.sort(v; by=abs, alg)
        end
        AK.sortperm(v; alg=SORT_ALG)
        AK.searchsortedfirst!(similar(v, Int), AK.sort(v), v)

        widened = String[]
        for name in names(AK; all=true)
            startswith(string(name), "gpu_") || continue
            for m in methods(getfield(AK, name)), mi in Base.specializations(m)
                Base.isdispatchtuple(mi.specTypes) || union!(widened, (string(name),))
            end
        end
        @test isempty(widened)
    end
end
