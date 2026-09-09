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

        # Stateful iterators must survive validation, including duplicate detection.
        expected = reverse(h; dims=(1, 3))
        v = array_from_host(h)
        @test AK.reverse!(v; dims=Iterators.Stateful([1, 3]), prefer_threads) === v
        @test Array(v) == expected
        src = array_from_host(h)
        @test Array(AK.reverse(src; dims=Iterators.Stateful([1, 3]), prefer_threads)) == expected
        dst = similar(src)
        @test AK.reverse!(dst, src; dims=Iterators.Stateful([1, 3]), prefer_threads) === dst
        @test Array(dst) == expected
        @test Array(src) == h

        # Destination assignment converts element types, as in the whole-array path.
        h_int = reshape(Int32.(1:30), 5, 6)
        src_int = array_from_host(h_int)
        for dims in (:, (), 1, (1, 2))
            dst_float = array_from_host(zeros(Float32, size(h_int)))
            @test AK.reverse!(dst_float, src_int; dims, prefer_threads) === dst_float
            @test Array(dst_float) == reverse(h_int; dims)
        end
        @test Array(src_int) == h_int

        # dims=() reverses nothing
        v = array_from_host(h)
        @test Array(AK.reverse!(v; dims=(), prefer_threads)) == h
        @test Array(AK.reverse(v; dims=(), prefer_threads)) == h

        # Shapes spanning many blocks, with odd extents so the in-place middle slice is
        # non-trivial, for integer element types too
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32)), shape in ((1001, 333), (33, 65, 129))
            h = rand(T, shape)
            for dims in (1, 2, (1, 2)), block_size in (64, 256)
                v = array_from_host(h)
                AK.reverse!(v; dims, prefer_threads, block_size)
                @test Array(v) == reverse(h; dims)

                out = AK.reverse(array_from_host(h); dims, prefer_threads, block_size)
                @test Array(out) == reverse(h; dims)
            end
        end

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

        dst = similar(v)
        @test_throws ArgumentError AK.reverse!(dst, v; dims=Iterators.Stateful([1, 1]), prefer_threads)
        @test_throws ArgumentError AK.reverse!(v; dims=nothing, prefer_threads)
        @test_throws ArgumentError AK.reverse!(similar(v, 4, 3, 2), v; dims=1, prefer_threads)

        # Duplicate dims throw, as in Base
        @test_throws ArgumentError AK.reverse!(v; dims=(1, 1), prefer_threads)
        @test_throws ArgumentError AK.reverse(v; dims=[2, 3, 2], prefer_threads)
    end
end
