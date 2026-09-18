@testset "sort_alg_kwarg" begin
    Random.seed!(2026)

    function is_valid_perm(vh, ixh; kwargs...)
        n = length(vh)
        length(ixh) == n &&
        sort(Int.(ixh)) == collect(1:n) &&
        issorted(vh[ixh]; kwargs...)
    end

    if !prefer_threads
        for T in valid_backend_eltypes(BACKEND,
                        (UInt32, Int32, Float32, UInt64, Int64, Float64))
            v_h = rand(T, 10_000)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort())
            @test Array(v) == sort(v_h)
        end

        v_h = rand(Int32, 10_000)
        v_default = array_from_host(v_h)
        v_merge = array_from_host(v_h)
        AK.sort!(v_default; prefer_threads)
        AK.sort!(v_merge; prefer_threads, alg=AK.MergeSort())
        @test Array(v_merge) == Array(v_default)

        perm_h = rand(Float32, 4096)
        for alg in (AK.MergeSort(), AK.MergeSort(lowmem=true))
            v = array_from_host(perm_h)
            ix = array_from_host(zeros(Int, length(perm_h)))
            temp = array_from_host(zeros(Int, length(perm_h)))
            AK.sortperm!(ix, v; prefer_threads, alg, temp)
            @test is_valid_perm(perm_h, Int.(Array(ix)))
        end

        v = array_from_host(rand(Float32, 128))
        ix = array_from_host(zeros(Int, length(v)))
        @test_throws ArgumentError AK.sort!(copy(v); prefer_threads, alg=AK.SampleSort())
        @test_throws ArgumentError AK.sortperm!(ix, v; prefer_threads, alg=AK.RadixSort())
    else
        v_h = rand(Int32, 10_000)
        v_default = array_from_host(v_h)
        v_sample = array_from_host(v_h)
        AK.sort!(v_default; prefer_threads)
        AK.sort!(v_sample; prefer_threads, alg=AK.SampleSort())
        @test Array(v_sample) == Array(v_default)

        ix = array_from_host(zeros(Int, length(v_h)))
        AK.sortperm!(ix, array_from_host(v_h); prefer_threads, alg=AK.SampleSort())
        @test is_valid_perm(v_h, Int.(Array(ix)))

        @test_throws ArgumentError AK.sort!(array_from_host(v_h); prefer_threads, alg=AK.MergeSort())
        @test_throws ArgumentError AK.sort!(array_from_host(v_h); prefer_threads, alg=AK.RadixSort())
        @test_throws ArgumentError AK.sortperm!(ix, array_from_host(v_h); prefer_threads, alg=AK.RadixSort())
    end
end
