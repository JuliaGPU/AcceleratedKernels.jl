

@testset "sort_dims" begin
    Random.seed!(0)

    # Fuzzy correctness against Base.sort(A; dims) for 2D and 3D arrays
    for _ in 1:100
        nd = rand(2:3)
        sz = ntuple(_ -> rand(1:15), nd)
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
            A_h = rand(T, sz...)
            A   = array_from_host(A_h)
            for dim in 1:nd, rev in (false, true)
                @test Array(AK.sort(A; prefer_threads, dims=dim, rev)) == sort(A_h; dims=dim, rev)
            end
        end
    end

    # Slice lengths around the block tile (2 * block_size) and beyond it, so that both the
    # block-level sort and the global merge passes are exercised, with duplicate-heavy data
    for len in (511, 512, 513, 1023, 1024, 1025, 2049, 10_000), nslices in (1, 3)
        A_h = rand(Int32(0):Int32(7), len, nslices)
        @test Array(AK.sort(array_from_host(A_h); prefer_threads, dims=1)) == sort(A_h; dims=1)
        @test Array(AK.sort(array_from_host(A_h); prefer_threads, dims=1, rev=true)) == sort(A_h; dims=1, rev=true)
        A_h = rand(Float32, nslices, len)
        @test Array(AK.sort(array_from_host(A_h); prefer_threads, dims=2)) == sort(A_h; dims=2)
    end

    # by, lt, order and temp act on the values within each slice
    A_h = rand(Float32, 300, 700)
    A   = array_from_host(A_h)
    @test Array(AK.sort(A; prefer_threads, dims=1, by=x->-x)) == sort(A_h; dims=1, by=x->-x)
    @test Array(AK.sort(A; prefer_threads, dims=2, lt=(>))) == sort(A_h; dims=2, lt=(>))
    @test Array(AK.sort(A; prefer_threads, dims=2, order=Base.Order.Reverse)) == sort(A_h; dims=2, order=Base.Order.Reverse)
    @test Array(AK.sort(A; prefer_threads, dims=2, temp=similar(A))) == sort(A_h; dims=2)
    if !prefer_threads
        @test Array(AK.sort(A; prefer_threads, dims=1, block_size=64)) == sort(A_h; dims=1)
        @test_throws ArgumentError AK.sort(A; prefer_threads, dims=1, alg=AK.RadixSort())
    end

    # NaNs, infinities and signed zeros order like Base
    A_h = Float32[NaN 1 -0.0; 0.0 -Inf NaN; 2 NaN Inf]
    A   = array_from_host(A_h)
    @test isequal(Array(AK.sort(A; prefer_threads, dims=1)), sort(A_h; dims=1))
    @test isequal(Array(AK.sort(A; prefer_threads, dims=2, rev=true)), sort(A_h; dims=2, rev=true))

    # In-place sorts each slice, leaves the array otherwise intact
    A_h = rand(Int32, 40, 31)
    A   = array_from_host(A_h)
    AK.sort!(A; prefer_threads, dims=2)
    @test Array(A) == sort(A_h; dims=2)

    # dims=1 on a vector is a full sort
    v_h = rand(Int32, 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sort(v; prefer_threads, dims=1)) == sort(v_h)

    # 4D arrays
    A_h = rand(Int32(0):Int32(3), 3, 4, 5, 6)
    A   = array_from_host(A_h)
    for dim in 1:4
        @test Array(AK.sort(A; prefer_threads, dims=dim)) == sort(A_h; dims=dim)
    end

    # Empty and singleton slices
    for sz in ((0, 5), (5, 0), (1, 64), (64, 1)), dim in 1:2
        A_h = rand(Float32, sz...)
        @test Array(AK.sort(array_from_host(A_h); prefer_threads, dims=dim)) == sort!(copy(A_h); dims=dim)
    end

    # Out-of-range dimension errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sort(A; prefer_threads, dims=3)
    @test_throws ArgumentError AK.sort(A; prefer_threads, dims=0)
end
