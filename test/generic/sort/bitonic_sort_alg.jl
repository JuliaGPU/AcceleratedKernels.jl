
@testset "bitonic_sort_alg" begin
    if !prefer_threads
        Random.seed!(0)
        alg = AK.BitonicSort()

        # Fuzz across element types, including ones RadixSort cannot handle
        for T in valid_backend_eltypes(BACKEND, (UInt8, Int16, Int32, UInt32, Float32, Int64, UInt64, Float64))
            for _ in 1:20
                n = rand(1:100_000)
                v_h = rand(T, n)
                v = array_from_host(v_h)
                AK.sort!(v; prefer_threads, alg)
                @test Array(v) == sort(v_h)
            end
        end

        # Lengths around the tile size (2048 by default) and through several global levels
        for n in (1, 2, 3, 7, 8, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096,
                  4097, 8191, 8192, 8193, 100_000, 1_000_000)
            v_h = rand(Float32, n)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg)
            @test Array(v) == sort(v_h)
        end

        # Adversarial patterns
        for n in (1000, 8192, 65536)
            for v_h in (fill(2.5f0, n), Float32.(1:n), Float32.(n:-1:1),
                        Float32.(rand(0:1, n)), Float32.(rand(0:3, n)))
                v = array_from_host(v_h)
                AK.sort!(v; prefer_threads, alg)
                @test Array(v) == sort(v_h)
            end
        end

        # NaNs and signed zeros follow `isless`, like Base
        v_h = rand(Float32, 5000)
        v_h[rand(1:5000, 100)] .= NaN32
        v_h[rand(1:5000, 100)] .= -0.0f0
        v_h[rand(1:5000, 100)] .= 0.0f0
        for rev in (false, true)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg, rev)
            @test isequal(Array(v), sort(v_h; rev))
        end

        # lt, by, rev and order
        v_h = rand(Int32, 10_000)
        for kw in ((rev=true,), (order=Base.Order.Reverse,), (rev=true, order=Base.Order.Reverse),
                   (lt=(>),), (by=abs,), (by=x -> x % Int32(7), rev=true), (lt=(a, b) -> a % 5 < b % 5,))
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg, kw...)
            sorted = Array(v)
            ord = Base.Order.ord(get(kw, :lt, isless), get(kw, :by, identity),
                                 get(kw, :rev, nothing), get(kw, :order, Base.Order.Forward))
            @test issorted(sorted; order=ord)
            @test sort(sorted) == sort(v_h)
        end

        # Tuning
        v_h = rand(UInt32, 20_000)
        for block_size in (32, 128, 256, 512), items_per_thread in (1, 2, 8, 16)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.BitonicSort(; block_size, items_per_thread))
            @test Array(v) == sort(v_h)
        end
        v = array_from_host(v_h)
        AK.sort!(v; prefer_threads, alg, block_size=64)
        @test Array(v) == sort(v_h)
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.BitonicSort(block_size=100))
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.BitonicSort(items_per_thread=3))
        @test_throws ArgumentError AK.sort!(v; prefer_threads,
            alg=AK.BitonicSort(block_size=2, items_per_thread=1 << (Sys.WORD_SIZE - 2)))

        # Empty input
        @test isempty(Array(AK.sort!(array_from_host(Int32[]); prefer_threads, alg)))

        # Matrices are sorted as one flat vector by default
        m_h = rand(Float32, 100, 30)
        m = array_from_host(m_h)
        AK.sort!(m; prefer_threads, alg)
        @test vec(Array(m)) == sort(vec(m_h))

        # Out-of-place: input unchanged
        v_h = rand(Float32, 10_000)
        v = array_from_host(v_h)
        w = AK.sort(v; prefer_threads, alg)
        @test Array(w) == sort(v_h)
        @test Array(v) == v_h

        # No permutation path
        @test_throws ArgumentError AK.sortperm(v; prefer_threads, alg)
    else
        @test_throws ArgumentError AK.sort!(rand(Int32, 16); prefer_threads, alg=AK.BitonicSort())
    end
end
