@testset "reverse" begin

    Random.seed!(0)

    # Sizes around the block boundary, plus the degenerate ones: an empty array, a
    # single element, and odd lengths whose middle element is its own mirror
    edge_sizes = [0, 1, 2, 3, 4, 5, 255, 256, 257, 511, 512, 513]

    @testset "reverse! in-place" begin
        for T in (Int32, Int64, Float32, Float64), n in edge_sizes
            h = rand(T, n)
            v = array_from_host(h)
            AK.reverse!(v; prefer_threads)
            @test Array(v) == reverse(h)
        end

        # Reversing twice restores the original
        for _ in 1:50
            h = rand(Float32, rand(1:100_000))
            v = array_from_host(h)
            AK.reverse!(v; prefer_threads)
            AK.reverse!(v; prefer_threads)
            @test Array(v) == h
        end

        # Returns the same array it was given, not a copy
        v = array_from_host(rand(Float32, 1000))
        @test AK.reverse!(v; prefer_threads) === v
    end

    @testset "reverse! out-of-place" begin
        for T in (Int32, Int64, Float32, Float64), n in edge_sizes
            h = rand(T, n)
            src = array_from_host(h)
            dst = array_from_host(zeros(T, n))
            AK.reverse!(dst, src; prefer_threads)
            @test Array(dst) == reverse(h)
            @test Array(src) == h                   # source left untouched
        end

        @test_throws Exception AK.reverse!(
            array_from_host(rand(Float32, 10)),
            array_from_host(rand(Float32, 11));
            prefer_threads,
        )
    end

    @testset "reverse allocating" begin
        for T in (Int32, Int64, Float32, Float64), n in edge_sizes
            h = rand(T, n)
            v = array_from_host(h)
            out = AK.reverse(v; prefer_threads)
            @test Array(out) == reverse(h)
            @test Array(v) == h                     # source left untouched
            @test out !== v
        end
    end

    # Randomised sweep over lengths that are not multiples of the block size
    @testset "random sizes" begin
        for _ in 1:100
            n = rand(1:100_000)
            h = rand(Float32, n)

            v = array_from_host(h)
            AK.reverse!(v; prefer_threads)
            @test Array(v) == reverse(h)

            src = array_from_host(h)
            dst = array_from_host(zeros(Float32, n))
            AK.reverse!(dst, src; prefer_threads)
            @test Array(dst) == reverse(h)
        end
    end

    # The tuning settings must not change results
    @testset "settings" begin
        h = rand(Float32, 10_000)
        for block_size in (32, 64, 128, 256)
            v = array_from_host(h)
            AK.reverse!(v; prefer_threads, block_size)
            @test Array(v) == reverse(h)
        end
        for (max_tasks, min_elems) in ((1, 1), (2, 100), (4, 1000))
            v = array_from_host(h)
            AK.reverse!(v; prefer_threads, max_tasks, min_elems)
            @test Array(v) == reverse(h)
        end
    end
end
