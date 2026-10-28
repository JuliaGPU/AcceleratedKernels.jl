# The tunings this backend's device returns resolve to algorithms that pass the checks, for
# `Auto()` and every algorithm the backend runs. Resolution only looks at types and sizes, so
# host arrays of the device's element types stand in for device arrays.

const TUNING_ELTYPES =
    valid_backend_eltypes(BACKEND, (UInt8, Int16, Int32, UInt32, Float32, Int64, Float64))
const FWD = Base.Order.Forward

@testset "tunings: sort" begin
    for T in TUNING_ELTYPES
        for len in (1, 100, 10_000, 10^7), stable in (true, false)
            v = Vector{T}(undef, len)
            @test AK.resolve_sort(AK.Auto(; stable), BACKEND, v, :, FWD) isa AK.SortAlgorithm
            @test AK.resolve_sort(AK.Auto(; stable), BACKEND, v, :, FWD; perm=true) isa
                  AK.SortAlgorithm
        end
        if AK.runs_kernels(BACKEND)
            for alg in (AK.MergeSort(), AK.BitonicSort(), AK.RadixSort())
                alg isa AK.RadixSort && !AK.radix_supports_eltype(T) && continue
                @test AK.resolve_sort(alg, BACKEND, zeros(T, 16), :, FWD) isa typeof(alg)
            end
        end
    end
end


@testset "tunings: reductions, scans, findall, any and all" begin
    kernels = AK.runs_kernels(BACKEND)
    for T in (Bool, TUNING_ELTYPES...)
        for dims in (:, (1,), (1, 2))
            @test AK.resolve_reduce(AK.Auto(), BACKEND, T, dims) isa AK.Algorithm
            kernels && @test AK.resolve_reduce(AK.BlockReduce(), BACKEND, T, dims) isa
                             AK.BlockReduce
        end
        for dims in (nothing, 1)
            @test AK.resolve_scan(AK.Auto(), BACKEND, T, dims) isa AK.Algorithm
            @test AK.resolve_scan(AK.Auto(), BACKEND, T, dims, typeof(AK.Optional{T}())) isa
                  AK.Algorithm
        end
        if kernels
            @test AK.resolve_scan(AK.ScanPrefixes(), BACKEND, T, nothing) isa AK.ScanPrefixes
            @test AK.resolve_scan(AK.SliceScan(), BACKEND, T, 1) isa AK.SliceScan
        end
        @test AK.resolve_findall(AK.Auto(), BACKEND, T) isa AK.Algorithm
        kernels && @test AK.resolve_findall(AK.ScanScatter(), BACKEND, T) isa AK.ScanScatter
        @test AK.resolve_predicate(AK.Auto(), BACKEND, T) isa AK.Algorithm
        if kernels
            @test AK.resolve_predicate(AK.ConcurrentWrite(), BACKEND, T) isa AK.ConcurrentWrite
            @test AK.resolve_predicate(AK.ViaReduce(), BACKEND, T) isa AK.ViaReduce
        end
    end
end
