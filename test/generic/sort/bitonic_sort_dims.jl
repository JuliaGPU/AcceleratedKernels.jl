
@testset "bitonic_sort_dims" begin
    if !prefer_threads
        Random.seed!(0)
        alg = AK.BitonicSort()

        # Slices that fit one tile and slices that need global passes, 2D and 3D
        for T in valid_backend_eltypes(BACKEND, (Int16, Int32, Float32, Int64, Float64))
            for (L, ncols) in ((8, 5), (256, 16), (1024, 4), (100, 50), (2048, 8), (2049, 3), (5000, 3), (20_000, 2))
                for dim in (1, 2)
                    sz = dim == 1 ? (L, ncols) : (ncols, L)
                    h = rand(T, sz...)
                    v = array_from_host(h)
                    AK.sort!(v; prefer_threads, dims=dim, alg)
                    @test Array(v) == sort(h; dims=dim)

                    v = array_from_host(h)
                    AK.sort!(v; prefer_threads, dims=dim, alg, rev=true)
                    @test Array(v) == sort(h; dims=dim, rev=true)
                end
            end

            h = rand(T, 7, 40, 5)
            for dim in (1, 2, 3)
                v = array_from_host(h)
                AK.sort!(v; prefer_threads, dims=dim, alg)
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
                AK.sort!(v; prefer_threads, dims=dim, alg=packed_alg, kw...)
                @test Array(v) == sort(h; dims=dim, kw...)
            end
        end

        # by / lt, NaNs, vectors, empty and singleton slices
        h = rand(Float32, 300, 700)
        @test Array(AK.sort(array_from_host(h); prefer_threads, dims=1, alg, by=x -> -x)) == sort(h; dims=1, by=x -> -x)
        @test Array(AK.sort(array_from_host(h); prefer_threads, dims=2, alg, lt=(>))) == sort(h; dims=2, lt=(>))
        h[rand(1:length(h), 1000)] .= NaN32
        @test isequal(Array(AK.sort(array_from_host(h); prefer_threads, dims=2, alg)), sort(h; dims=2))
        h = rand(Float32, 1000)
        @test Array(AK.sort(array_from_host(h); prefer_threads, dims=1, alg)) == sort(h)
        @test size(AK.sort(array_from_host(rand(Float32, 0, 5)); prefer_threads, dims=1, alg)) == (0, 5)
        @test size(AK.sort(array_from_host(rand(Float32, 5, 0)); prefer_threads, dims=1, alg)) == (5, 0)
        h = rand(Float32, 1, 100)
        @test Array(AK.sort(array_from_host(h); prefer_threads, dims=1, alg)) == h

        # Tuning applies per slice; out-of-place leaves the input untouched
        h = rand(Float32, 3000, 10)
        for items_per_thread in (1, 4, 16)
            v = array_from_host(h)
            AK.sort!(v; prefer_threads, dims=1, alg=AK.BitonicSort(; block_size=128, items_per_thread))
            @test Array(v) == sort(h; dims=1)
        end
        v = array_from_host(h)
        w = AK.sort(v; prefer_threads, dims=1, alg)
        @test Array(w) == sort(h; dims=1)
        @test Array(v) == h

        @test_throws ArgumentError AK.sort!(array_from_host(h); prefer_threads, dims=3, alg)
    end
end
