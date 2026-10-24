@testset "sort: dims" begin
    Random.seed!(0)

    # Fuzzy correctness against Base.sort(A; dims) for 2D and 3D arrays
    for _ in 1:100
        nd = rand(2:3)
        sz = ntuple(_ -> rand(1:15), nd)
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
            A_h = rand(T, sz...)
            A   = array_from_host(A_h)
            for dim in 1:nd, rev in (false, true)
                @test Array(AK.sort(A; alg=SORT_ALG, dims=dim, rev)) == sort(A_h; dims=dim, rev)
            end
        end
    end

    # Slice lengths around the block tile (2 * block_size) and beyond it, so that both the
    # block-level sort and the global merge passes are exercised, with duplicate-heavy data
    for len in (511, 512, 513, 1023, 1024, 1025, 2049, 10_000), nslices in (1, 3)
        A_h = rand(Int32(0):Int32(7), len, nslices)
        @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=1)) == sort(A_h; dims=1)
        @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=1, rev=true)) == sort(A_h; dims=1, rev=true)
        A_h = rand(Float32, nslices, len)
        @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=2)) == sort(A_h; dims=2)
    end

    # by, lt, order and a workspace act on the values within each slice
    A_h = rand(Float32, 300, 700)
    A   = array_from_host(A_h)
    @test Array(AK.sort(A; alg=SORT_ALG, dims=1, by=x->-x)) == sort(A_h; dims=1, by=x->-x)
    @test Array(AK.sort(A; alg=SORT_ALG, dims=2, lt=(>))) == sort(A_h; dims=2, lt=(>))
    @test Array(AK.sort(A; alg=SORT_ALG, dims=2, order=Base.Order.Reverse)) == sort(A_h; dims=2, order=Base.Order.Reverse)
    @test Array(with_workspace(AK.sort, A; alg=SORT_ALG, dims=2)) == sort(A_h; dims=2)
    if TEST_KERNELS
        @test Array(AK.sort(A; alg=AK.MergeSort(block_size=64), dims=1)) == sort(A_h; dims=1)
    end

    # NaNs, infinities and signed zeros order like Base
    A_h = Float32[NaN 1 -0.0; 0.0 -Inf NaN; 2 NaN Inf]
    A   = array_from_host(A_h)
    @test isequal(Array(AK.sort(A; alg=SORT_ALG, dims=1)), sort(A_h; dims=1))
    @test isequal(Array(AK.sort(A; alg=SORT_ALG, dims=2, rev=true)), sort(A_h; dims=2, rev=true))

    # In-place sorts each slice, leaves the array otherwise intact
    A_h = rand(Int32, 40, 31)
    A   = array_from_host(A_h)
    AK.sort!(A; alg=SORT_ALG, dims=2)
    @test Array(A) == sort(A_h; dims=2)

    # dims=1 on a vector is a full sort
    v_h = rand(Int32, 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sort(v; alg=SORT_ALG, dims=1)) == sort(v_h)

    # 4D arrays
    A_h = rand(Int32(0):Int32(3), 3, 4, 5, 6)
    A   = array_from_host(A_h)
    for dim in 1:4
        @test Array(AK.sort(A; alg=SORT_ALG, dims=dim)) == sort(A_h; dims=dim)
    end

    # Slices after dimensions of size 1 are contiguous, like those along the first dimension
    A_h = rand(Int32, 1, 3000, 7)
    @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=2)) == sort(A_h; dims=2)
    A_h = rand(Float32, 1, 1, 300, 5)
    @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=3, rev=true)) ==
          sort(A_h; dims=3, rev=true)

    # Empty and singleton slices
    for sz in ((0, 5), (5, 0), (1, 64), (64, 1)), dim in 1:2
        A_h = rand(Float32, sz...)
        @test Array(AK.sort(array_from_host(A_h); alg=SORT_ALG, dims=dim)) == sort!(copy(A_h); dims=dim)
    end

    # Out-of-range dimension errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sort(A; alg=SORT_ALG, dims=3)
    @test_throws ArgumentError AK.sort(A; alg=SORT_ALG, dims=0)
end


if TEST_KERNELS
@testset "sort: BitonicSort along dims" begin
    Random.seed!(0)
    alg = AK.BitonicSort()

    # Slices that fit one tile and slices that need global passes, 2D and 3D
    for T in valid_backend_eltypes(BACKEND, (Int16, Int32, Float32, Int64, Float64))
        for (L, ncols) in ((8, 5), (256, 16), (1024, 4), (100, 50), (2048, 8), (2049, 3), (5000, 3), (20_000, 2))
            for dim in (1, 2)
                sz = dim == 1 ? (L, ncols) : (ncols, L)
                h = rand(T, sz...)
                v = array_from_host(h)
                AK.sort!(v; dims=dim, alg)
                @test Array(v) == sort(h; dims=dim)

                v = array_from_host(h)
                AK.sort!(v; dims=dim, alg, rev=true)
                @test Array(v) == sort(h; dims=dim, rev=true)
            end
        end

        h = rand(T, 7, 40, 5)
        for dim in (1, 2, 3)
            v = array_from_host(h)
            AK.sort!(v; dims=dim, alg)
            @test Array(v) == sort(h; dims=dim)
        end
    end

    # Packed slices spanning two tiles, with a partial final tile and captured orderings
    packed_alg = AK.BitonicSort(block_size=8, items_per_thread=4)
    mask = UInt32(0x55)
    for len in (3, 4), dim in (1, 2)
        h = rand(UInt32, dim == 1 ? (len, 9) : (9, len))
        for kw in ((by=x -> xor(x, mask),), (lt=(a, b) -> xor(a, mask) < xor(b, mask),))
            v = array_from_host(h)
            AK.sort!(v; dims=dim, alg=packed_alg, kw...)
            @test Array(v) == sort(h; dims=dim, kw...)
        end
    end

    # by / lt, NaNs, vectors, empty and singleton slices
    h = rand(Float32, 300, 700)
    @test Array(AK.sort(array_from_host(h); dims=1, alg, by=x -> -x)) == sort(h; dims=1, by=x -> -x)
    @test Array(AK.sort(array_from_host(h); dims=2, alg, lt=(>))) == sort(h; dims=2, lt=(>))
    h[rand(1:length(h), 1000)] .= NaN32
    @test isequal(Array(AK.sort(array_from_host(h); dims=2, alg)), sort(h; dims=2))
    h = rand(Float32, 1000)
    @test Array(AK.sort(array_from_host(h); dims=1, alg)) == sort(h)
    @test size(AK.sort(array_from_host(rand(Float32, 0, 5)); dims=1, alg)) == (0, 5)
    @test size(AK.sort(array_from_host(rand(Float32, 5, 0)); dims=1, alg)) == (5, 0)
    h = rand(Float32, 1, 100)
    @test Array(AK.sort(array_from_host(h); dims=1, alg)) == h

    # Tuning applies per slice; out-of-place leaves the input untouched
    h = rand(Float32, 3000, 10)
    for items_per_thread in (1, 4, 16)
        v = array_from_host(h)
        AK.sort!(v; dims=1, alg=AK.BitonicSort(; block_size=128, items_per_thread))
        @test Array(v) == sort(h; dims=1)
    end
    v = array_from_host(h)
    w = AK.sort(v; dims=1, alg)
    @test Array(w) == sort(h; dims=1)
    @test Array(v) == h

    @test_throws ArgumentError AK.sort!(array_from_host(h); dims=3, alg)
end
end


if TEST_KERNELS
@testset "sort: RadixSort along dims" begin
    Random.seed!(0)
    alg = AK.RadixSort()

    # Slices sorted in local memory (up to 2 * block_size = 512 elements) and by the passes
    # over all slices, 2D and 3D
    for T in valid_backend_eltypes(BACKEND, (UInt32, Int32, Float32, UInt64, Int64, Float64))
        for (L, ncols) in ((8, 5), (256, 16), (512, 4), (513, 4), (100, 50), (2049, 3),
                           (20_000, 2), (300_000, 2))
            for dim in (1, 2)
                sz = dim == 1 ? (L, ncols) : (ncols, L)
                h = T <: AbstractFloat ? rand(T, sz...) : rand(T(1):T(1000), sz...)
                @test Array(AK.sort(array_from_host(h); dims=dim, alg)) == sort(h; dims=dim)
                @test Array(AK.sort(array_from_host(h); dims=dim, alg, rev=true)) ==
                      sort(h; dims=dim, rev=true)
            end
        end

        h = rand(T, 7, 600, 5)
        for dim in (1, 2, 3)
            @test Array(AK.sort(array_from_host(h); dims=dim, alg)) == sort(h; dims=dim)
        end
        h = rand(T, 1, 3000, 3)             # contiguous slices after a singleton dimension
        @test Array(AK.sort(array_from_host(h); dims=2, alg)) == sort(h; dims=2)
    end

    # Signed zeros, infinities and NaNs order like Base
    for L in (300, 3000)
        h = rand(Float32[-0.0, 0.0, 1, -Inf, Inf, NaN], L, 7)
        for dim in (1, 2), rev in (false, true)
            @test isequal(Array(AK.sort(array_from_host(h); dims=dim, alg, rev)),
                          sort(h; dims=dim, rev))
        end
    end

    # The chunked and portable kernels (a block size that is not a multiple of 32), and
    # several items per thread
    h = rand(Int32, 3000, 10)
    for alg in (AK.RadixSort(block_size=16), AK.RadixSort(block_size=128, items_per_thread=8),
                AK.RadixSort(chunk_size=64))
        @test Array(AK.sort(array_from_host(h); dims=1, alg)) == sort(h; dims=1)
        @test Array(AK.sort(array_from_host(h); dims=2, alg)) == sort(h; dims=2)
    end

    # Vectors, empty and singleton slices, workspaces, out-of-place
    h = rand(Float32, 1000)
    @test Array(AK.sort(array_from_host(h); dims=1, alg)) == sort(h)
    @test size(AK.sort(array_from_host(rand(Float32, 0, 5)); dims=1, alg)) == (0, 5)
    @test size(AK.sort(array_from_host(rand(Float32, 5, 0)); dims=1, alg)) == (5, 0)
    h = rand(Float32, 1, 100)
    @test Array(AK.sort(array_from_host(h); dims=1, alg)) == h
    h = rand(Float32, 3000, 10)
    v = array_from_host(h)
    @test Array(with_workspace(AK.sort!, v; dims=2, alg)) == sort(h; dims=2)
    v = array_from_host(h)
    w = AK.sort(v; dims=1, alg)
    @test Array(w) == sort(h; dims=1)
    @test Array(v) == h

    @test_throws ArgumentError AK.sort!(array_from_host(h); dims=3, alg)
end
end
