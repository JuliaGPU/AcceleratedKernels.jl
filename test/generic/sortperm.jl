# The permutation algorithms (`Auto()` always picks the merge sort on GPUs)
PERM_ALGS = TEST_KERNELS ? [SORT_ALG, AK.MergeSort(lowmem=true)] : [SORT_ALG]


@testset "sort_by_key" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Int32, num_elems))
        v = copy(k) .- 1
        AK.sort_by_key!(k, v; alg=SORT_ALG)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(UInt32, num_elems))
        v = copy(k) .- 1
        AK.sort_by_key!(k, v; alg=SORT_ALG)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Float32, num_elems))
        v = copy(k) .- 1
        AK.sort_by_key!(k, v; alg=SORT_ALG)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    # Testing different settings
    alg = sort_alg(block_size=64, max_tasks=64, min_elems=8)
    k = array_from_host(1:10_000, Float32)
    v = array_from_host(1:10_000, Int32)
    with_workspace(AK.sort_by_key!, k, v; lt=(>), by=abs, rev=true, alg)
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    with_workspace(AK.sort_by_key!, k, v; lt=(>), by=abs, rev=true, alg)
    @test issorted(Array(k))
    @test issorted(Array(v))

    # Stable: the values of equal keys keep their order, whole arrays and along `dims`, including
    # a sample sort that runs several tasks
    kh = rand(Int32(1):Int32(20), 50_000)
    k = array_from_host(kh)
    v = array_from_host(Int32.(1:50_000))
    AK.sort_by_key!(k, v; alg=SORT_ALG)
    @test Array(k) == sort(kh)
    @test Array(v) == sortperm(kh)
    if AK.runs_threads(BACKEND)
        k = array_from_host(kh)
        v = array_from_host(Int32.(1:50_000))
        AK.sort_by_key!(k, v; alg=AK.CPUThreads.SampleSort(max_tasks=4))
        @test Array(v) == sortperm(kh)
    end
    Kh = rand(Int32(1):Int32(5), 300, 7)
    for dims in (1, 2)
        k = array_from_host(Kh)
        v = array_from_host(reshape(Int32.(1:length(Kh)), size(Kh)))
        AK.sort_by_key!(k, v; alg=SORT_ALG, dims)
        @test Array(k) == sort(Kh; dims)
        @test Array(v) == sortperm(Kh; dims)
    end

    # Arrays of any dimension, sorted as one flat vector, small and large
    for sz in ((2, 2), (3, 4, 5), (300, 70))
        Kh = rand(Int32(1):Int32(5), sz)
        k = array_from_host(Kh)
        v = array_from_host(reshape(Int32.(1:length(Kh)), sz))
        AK.sort_by_key!(k, v; alg=SORT_ALG)
        @test vec(Array(k)) == sort(vec(Kh))
        @test vec(Array(v)) == sortperm(vec(Kh))
        A = array_from_host(Kh)
        @test vec(Array(AK.sort!(A; alg=SORT_ALG))) == sort(vec(Kh))
        ix = array_from_host(zeros(Int, sz))
        @test vec(Array(AK.sortperm!(ix, array_from_host(Kh); alg=SORT_ALG))) == sortperm(vec(Kh))
    end

    # A workspace for other sizes is rejected before anything is modified
    kh = rand(Int32, 2000)
    k, v = array_from_host(kh), array_from_host(rand(Int32, 2000))
    @test_throws ArgumentError AK.sort_by_key!(k, v; alg=SORT_ALG, workspace=AK.workspace(
        AK.sort_by_key!, array_from_host(rand(Int32, 4000)), array_from_host(rand(Int32, 4000));
        alg=SORT_ALG))
    @test Array(k) == kh

    # Mismatched sizes
    @test_throws ArgumentError AK.sort_by_key!(array_from_host(rand(Int32, 10)),
                                               array_from_host(rand(Int32, 9)); alg=SORT_ALG)
end


@testset "sortperm: random inputs" begin
    Random.seed!(0)
    for alg in PERM_ALGS, T in (Int32, UInt32, Float32), _ in 1:1000
        n = rand(1:100_000)
        vh = rand(T, n)
        ix = array_from_host(zeros(Int32, n))
        AK.sortperm!(ix, array_from_host(vh); alg)
        ixh = Array(ix)
        @test isperm(ixh) && issorted(vh[ixh])
    end
    for T in valid_backend_eltypes(BACKEND, (Int16, UInt16, Int64, UInt64, Float64, UInt8)),
        _ in 1:50
        n = rand(1:50_000)
        vh = rand(T, n)
        ix = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, array_from_host(vh); alg=SORT_ALG)
        ixh = Array(ix)
        @test isperm(ixh) && issorted(vh[ixh])
    end
end


@testset "sortperm: settings" begin
    Random.seed!(0)
    algs = TEST_KERNELS ? [sort_alg(block_size=64), AK.MergeSort(lowmem=true, block_size=64)] :
                         [sort_alg(max_tasks=64, min_elems=8)]
    for alg in algs
        ix = array_from_host(1:10_000, Int32)
        v = array_from_host(1:10_000, Float32)
        with_workspace(AK.sortperm!, ix, v; alg, lt=(>), by=abs, rev=true)
        @test issorted(Array(v)[Array(ix)])

        vh = rand(Float32, 10_000)
        v = array_from_host(vh)
        ix = with_workspace(AK.sortperm, v; alg, lt=(>), by=abs, rev=true)
        @test issorted(vh[Array(ix)]; lt=(>), by=abs, rev=true)
        @test Array(v) == vh
    end

    # Base's permutation, as the sort is stable: for edge lengths, orderings, sorted and reversed
    # inputs and ties
    alg = SORT_ALG
    for n in (1, 2, 3, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049)
        vh = rand(Float32, n)
        @test Array(AK.sortperm(array_from_host(vh); alg)) == sortperm(vh)
    end
    base = rand(Float32, 2^14)
    cases = Any[sort(base), reverse(sort(base)), fill(1.0f0, 2^14), Float32.(rand(1:4, 2^14)),
                Int32.(mod.(1:10_000, 10))]
    for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
        push!(cases, rand(T, 10_000))
    end
    for vh in cases
        @test Array(AK.sortperm(array_from_host(vh); alg)) == sortperm(vh)
    end
    vh = randn(Float32, 10_000)
    for kw in ((rev=true,), (by=abs,), (by=abs, rev=true), (lt=(>),))
        @test Array(AK.sortperm(array_from_host(vh); alg, kw...)) == sortperm(vh; kw...)
    end

    # Keys `by.(v)` of another type than `v`, with ties, and of an array without a backend
    vh = rand(Int32(1):Int32(100), 10_000)
    by = x -> Float32(x % 7)
    @test Array(with_workspace(AK.sortperm, array_from_host(vh); by, alg)) == sortperm(vh; by)
    @test Array(AK.sortperm(1:1000; backend=BACKEND, by=x -> x % 10, alg)) ==
          sortperm(1:1000; by=x -> x % 10)
end


@testset "sortperm: dims" begin
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
                ix = Array(AK.sortperm(A; alg=SORT_ALG, dims=dim, rev))
                @test ix == sortperm(A_h; dims=dim, rev)
                @test A_h[ix] == sort(A_h; dims=dim, rev)
            end
        end
    end

    # Ties across block tiles and global merge passes must stay stable
    for len in (511, 512, 513, 1025, 2049, 10_000), nslices in (1, 3)
        A_h = rand(Int32(0):Int32(3), len, nslices)
        @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=1)) == sortperm(A_h; dims=1)
        @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=1, rev=true)) == sortperm(A_h; dims=1, rev=true)
        A_h = rand(Int32(0):Int32(3), nslices, len)
        @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=2)) == sortperm(A_h; dims=2)
    end

    # by, order, a workspace and the low-memory GPU path
    A_h = rand(Float32, 300, 700)
    A   = array_from_host(A_h)
    @test Array(AK.sortperm(A; alg=SORT_ALG, dims=1, by=x->-x)) == sortperm(A_h; dims=1, by=x->-x)
    @test Array(AK.sortperm(A; alg=SORT_ALG, dims=2, order=Base.Order.Reverse)) == sortperm(A_h; dims=2, order=Base.Order.Reverse)
    @test Array(with_workspace(AK.sortperm, A; alg=SORT_ALG, dims=2)) == sortperm(A_h; dims=2)
    if TEST_KERNELS
        @test Array(AK.sortperm(A; dims=2, alg=AK.MergeSort(lowmem=true))) == sortperm(A_h; dims=2)
        @test Array(AK.sortperm(A; dims=1, alg=AK.MergeSort(lowmem=true, block_size=64))) == sortperm(A_h; dims=1)
    end

    # In-place fills ix with the same global linear indices as Base
    A_h = rand(Int32(0):Int32(5), 40, 31)
    A   = array_from_host(A_h)
    ix  = array_from_host(zeros(Int, 40, 31))
    AK.sortperm!(ix, A; alg=SORT_ALG, dims=2)
    @test Array(ix) == sortperm(A_h; dims=2)

    # dims=1 on a vector is a full sortperm
    v_h = rand(Int32(0):Int32(9), 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sortperm(v; alg=SORT_ALG, dims=1)) == sortperm(v_h)

    # Slices after dimensions of size 1 are contiguous, like those along the first dimension
    A_h = rand(Int32(0):Int32(9), 1, 3000, 7)
    @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=2)) == sortperm(A_h; dims=2)
    A_h = rand(Float32, 1, 1, 300, 5)
    @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=3, rev=true)) ==
          sortperm(A_h; dims=3, rev=true)

    # Empty and singleton slices
    for sz in ((0, 5), (5, 0), (1, 64), (64, 1)), dim in 1:2
        A_h = rand(Float32, sz...)
        @test Array(AK.sortperm(array_from_host(A_h); alg=SORT_ALG, dims=dim)) == sortperm(A_h; dims=dim)
    end

    # Slice offsets refer to the view's linear indexing, not its parent's strides.
    A_h = rand(Int32(0):Int32(3), 6, 1030)
    A = array_from_host(A_h)
    V_h = view(A_h, 1:2:6, 1:2:1030)
    V = view(A, 1:2:6, 1:2:1030)
    @test Array(AK.sortperm(V; alg=SORT_ALG, dims=2)) == sortperm(V_h; dims=2)
    AK.sort!(V; alg=SORT_ALG, dims=2)
    sort!(V_h; dims=2)
    @test Array(A) == A_h

    # Floating-point ordering also applies to the low-level entry points.
    A_h = Float32[NaN 1 -0.0; 0.0 -Inf NaN; 2 NaN Inf]
    A = array_from_host(A_h)
    for dim in 1:2, rev in (false, true)
        expected = sortperm(A_h; dims=dim, rev)
        @test Array(AK.sortperm(A; alg=SORT_ALG, dims=dim, rev)) == expected
        if TEST_KERNELS
            @test Array(AK.sortperm(A; alg=AK.MergeSort(), dims=dim, rev)) == expected
            @test Array(AK.sortperm(A; alg=AK.MergeSort(lowmem=true), dims=dim, rev)) == expected
        end
    end

    # Invalid dimensions must not overwrite the output.
    ix = array_from_host(fill(-1, size(A_h)))
    @test_throws ArgumentError AK.sortperm!(ix, A; alg=SORT_ALG, dims=3)
    @test Array(ix) == fill(-1, size(A_h))
    if TEST_KERNELS
        @test_throws ArgumentError AK.sortperm!(ix, A; alg=AK.MergeSort(), dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.sortperm!(ix, A; alg=AK.MergeSort(lowmem=true), dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.sort_by_key!(copy(A), similar(ix, length(ix)); alg=AK.MergeSort(), dims=1)
    end

    # Out-of-range dimension and mismatched index array errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sortperm(A; alg=SORT_ALG, dims=3)
    @test_throws ArgumentError AK.sortperm(A; alg=SORT_ALG, dims=0)
    @test_throws ArgumentError AK.sortperm!(array_from_host(zeros(Int, 64)), A; alg=SORT_ALG, dims=1)
end
