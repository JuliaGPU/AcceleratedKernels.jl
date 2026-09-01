@testset "reverse" begin

    Random.seed!(0)

    # Sizes around the block boundary, plus the degenerate ones: an empty array, a
    # single element, and odd lengths whose middle element is its own mirror
    edge_sizes = [0, 1, 2, 3, 4, 5, 255, 256, 257, 511, 512, 513]

    # For backends that don't support Float64.
    test_types = valid_backend_eltypes(BACKEND,
                        (Int8, UInt32, Int64, Float32, Float64))

    @testset "reverse! in-place" begin
        for T in test_types, n in edge_sizes
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
        for T in test_types, n in edge_sizes
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
        for T in test_types, n in edge_sizes
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

    # Contiguous linear sub-range via start/stop (Base.reverse!/reverse (v, start, stop) parity)
    @testset "start/stop sub-range" begin
        for T in test_types, n in (1, 2, 5, 256, 257, 1000)
            h = rand(T, n)

            # a spread of ranges: whole, prefix, suffix, interior, single, empty (start > stop)
            ranges = [(1, n), (1, n ÷ 2 + 1), (n ÷ 2 + 1, n),
                      (max(1, n ÷ 4), min(n, 3n ÷ 4)), (min(n, 2), min(n, 2)),
                      (min(n, 3), min(n, 2))]
            for (lo, hi) in ranges
                # keep only in-bounds ranges (or empty ones, start > stop, which are no-ops)
                lo <= hi && !(1 <= lo && hi <= n) && continue

                # in-place
                v = array_from_host(h)
                AK.reverse!(v; start=lo, stop=hi, prefer_threads)
                @test Array(v) == reverse!(copy(h), lo, hi)

                # allocating out-of-place: source untouched, rest copied verbatim
                src = array_from_host(h)
                out = AK.reverse(src; start=lo, stop=hi, prefer_threads)
                @test Array(out) == reverse(h, lo, hi)
                @test Array(src) == h
                @test out !== src

                # out-of-place into dst
                dst = array_from_host(zeros(T, n))
                AK.reverse!(dst, src; start=lo, stop=hi, prefer_threads)
                @test Array(dst) == reverse(h, lo, hi)
                @test Array(src) == h
            end
        end

        # start/stop and dims are mutually exclusive
        m = array_from_host(rand(Float32, 4, 5))
        @test_throws ArgumentError AK.reverse!(m; dims=1, start=2, prefer_threads)
        @test_throws ArgumentError AK.reverse(m; dims=2, stop=3, prefer_threads)

        # out-of-bounds sub-range throws
        v = array_from_host(rand(Float32, 10))
        @test_throws BoundsError AK.reverse!(v; start=0, stop=5, prefer_threads)
        @test_throws BoundsError AK.reverse!(v; start=3, stop=11, prefer_threads)
    end

    # N-dimensional reversal along a subset of dimensions (Base.reverse parity)
    @testset "dims" begin
        # Single dimension, including a degenerate size-1 dim and a large 3-D array
        for shape in ([1, 2, 4, 3], [4, 2], [5], [8, 8, 8]),
            dim in 1:length(shape)

            h = rand(Float32, shape...)

            v = array_from_host(h)
            AK.reverse!(v; dims=dim, prefer_threads)
            @test Array(v) == reverse(h; dims=dim)

            src = array_from_host(h)
            out = AK.reverse(src; dims=dim, prefer_threads)
            @test Array(out) == reverse(h; dims=dim)
            @test Array(src) == h                       # source left untouched

            dst = array_from_host(zeros(Float32, shape...))
            AK.reverse!(dst, src; dims=dim, prefer_threads)
            @test Array(dst) == reverse(h; dims=dim)
        end

        # Multiple dimensions at once, plus dims=: (dispatches to the flat whole-array path).
        # The odd sizes of [7, 6, 5] exercise the in-place middle-plane swaps, where only the
        # index ordering guard stops a pair from being swapped twice
        for shape in ([1, 2, 4, 3], [8, 8, 8], [7, 6, 5]),
            dims in ((1, 2), (2, 3), (1, 3), :)

            h = rand(Float32, shape...)

            v = array_from_host(h)
            AK.reverse!(v; dims=dims, prefer_threads)
            @test Array(v) == reverse(h; dims=dims)

            out = AK.reverse(array_from_host(h); dims=dims, prefer_threads)
            @test Array(out) == reverse(h; dims=dims)

            src = array_from_host(h)
            dst = array_from_host(zeros(Float32, shape...))
            AK.reverse!(dst, src; dims=dims, prefer_threads)
            @test Array(dst) == reverse(h; dims=dims)
        end

        # Any iterable of integers works, e.g. a Vector (Base only accepts tuples)
        h = rand(Float32, 4, 5, 6)
        out = AK.reverse(array_from_host(h); dims=[1, 3], prefer_threads)
        @test Array(out) == reverse(h; dims=(1, 3))

        # Empty arrays are returned unchanged
        h = zeros(Float32, 0, 5)
        for dims in (1, 2, (1, 2))
            v = array_from_host(h)
            @test Array(AK.reverse!(v; dims, prefer_threads)) == reverse(h; dims)

            dst = array_from_host(copy(h))
            @test Array(AK.reverse!(dst, v; dims, prefer_threads)) == reverse(h; dims)

            @test Array(AK.reverse(v; dims, prefer_threads)) == reverse(h; dims)
        end
    end

    # Invalid dims arguments throw, matching Base/CUDA
    @testset "dims errors" begin
        v = array_from_host(rand(Float32, 2, 3, 4))
        @test_throws ArgumentError AK.reverse!(v; dims=0, prefer_threads)
        @test_throws ArgumentError AK.reverse!(v; dims=4, prefer_threads)
        @test_throws ArgumentError AK.reverse(v; dims=0, prefer_threads)
        @test_throws ArgumentError AK.reverse(v; dims=4, prefer_threads)

        # Non-integer dims must throw rather than silently do nothing
        @test_throws ArgumentError AK.reverse!(v; dims=1.5, prefer_threads)
        @test_throws ArgumentError AK.reverse(v; dims=(1, 2.5), prefer_threads)
    end
end
