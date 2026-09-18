
@testset "sortperm_dims" begin
    Random.seed!(0)

    # Fuzzy correctness against Base.sortperm(A; dims); small integer ranges give many ties, so
    # matching Base's index array exactly also checks that the permutation is stable
    for _ in 1:100
        nd = rand(2:3)
        sz = ntuple(_ -> rand(1:15), nd)
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
            A_h = T <: Integer ? rand(T(0):T(4), sz...) : rand(T, sz...)
            A   = array_from_host(A_h)
            for dim in 1:nd, rev in (false, true)
                ix = Array(AK.sortperm(A; prefer_threads, dims=dim, rev))
                @test ix == sortperm(A_h; dims=dim, rev)
                @test A_h[ix] == sort(A_h; dims=dim, rev)
            end
        end
    end

    # Ties across block tiles and global merge passes must stay stable
    for len in (511, 512, 513, 1025, 2049, 10_000), nslices in (1, 3)
        A_h = rand(Int32(0):Int32(3), len, nslices)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=1)) == sortperm(A_h; dims=1)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=1, rev=true)) == sortperm(A_h; dims=1, rev=true)
        A_h = rand(Int32(0):Int32(3), nslices, len)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=2)) == sortperm(A_h; dims=2)
    end

    # by, order, temp and the low-memory GPU path
    A_h = rand(Float32, 300, 700)
    A   = array_from_host(A_h)
    @test Array(AK.sortperm(A; prefer_threads, dims=1, by=x->-x)) == sortperm(A_h; dims=1, by=x->-x)
    @test Array(AK.sortperm(A; prefer_threads, dims=2, order=Base.Order.Reverse)) == sortperm(A_h; dims=2, order=Base.Order.Reverse)
    @test Array(AK.sortperm(A; prefer_threads, dims=2, temp=similar(A, Int))) == sortperm(A_h; dims=2)
    if !prefer_threads
        @test Array(AK.sortperm(A; prefer_threads, dims=2, alg=AK.MergeSort(lowmem=true))) == sortperm(A_h; dims=2)
        @test Array(AK.sortperm(A; prefer_threads, dims=1, alg=AK.MergeSort(lowmem=true), block_size=64)) == sortperm(A_h; dims=1)
    end

    # In-place fills ix with the same global linear indices as Base
    A_h = rand(Int32(0):Int32(5), 40, 31)
    A   = array_from_host(A_h)
    ix  = array_from_host(zeros(Int, 40, 31))
    AK.sortperm!(ix, A; prefer_threads, dims=2)
    @test Array(ix) == sortperm(A_h; dims=2)

    # dims=1 on a vector is a full sortperm
    v_h = rand(Int32(0):Int32(9), 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sortperm(v; prefer_threads, dims=1)) == sortperm(v_h)

    # Empty and singleton slices
    for sz in ((0, 5), (5, 0), (1, 64), (64, 1)), dim in 1:2
        A_h = rand(Float32, sz...)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=dim)) == sortperm(A_h; dims=dim)
    end

    # Slice offsets refer to the view's linear indexing, not its parent's strides.
    A_h = rand(Int32(0):Int32(3), 6, 1030)
    A = array_from_host(A_h)
    V_h = view(A_h, 1:2:6, 1:2:1030)
    V = view(A, 1:2:6, 1:2:1030)
    @test Array(AK.sortperm(V; prefer_threads, dims=2)) == sortperm(V_h; dims=2)
    AK.sort!(V; prefer_threads, dims=2)
    sort!(V_h; dims=2)
    @test Array(A) == A_h

    # Floating-point ordering also applies to the low-level entry points.
    A_h = Float32[NaN 1 -0.0; 0.0 -Inf NaN; 2 NaN Inf]
    A = array_from_host(A_h)
    for dim in 1:2, rev in (false, true)
        expected = sortperm(A_h; dims=dim, rev)
        @test Array(AK.sortperm(A; prefer_threads, dims=dim, rev)) == expected
        if !prefer_threads
            @test Array(AK.merge_sortperm(A; dims=dim, rev)) == expected
            @test Array(AK.merge_sortperm_lowmem(A; dims=dim, rev)) == expected
        end
    end

    # Invalid dimensions must not overwrite the output.
    ix = array_from_host(fill(-1, size(A_h)))
    @test_throws ArgumentError AK.sortperm!(ix, A; prefer_threads, dims=3)
    @test Array(ix) == fill(-1, size(A_h))
    if !prefer_threads
        @test_throws ArgumentError AK.merge_sortperm!(ix, A; dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.merge_sortperm_lowmem!(ix, A; dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.merge_sort_by_key!(copy(A), similar(ix, length(ix)); dims=1)
    end

    # Out-of-range dimension and mismatched index array errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=3)
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=0)
    @test_throws ArgumentError AK.sortperm!(array_from_host(zeros(Int, 64)), A; prefer_threads, dims=1)
end
