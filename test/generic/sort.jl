# The sorting algorithms of the configuration, with the element types and the number of random
# inputs each sorts. `Auto()` picks among them by the device's tuning, and on the host always the
# sample sort.
SORT_FUZZ = TEST_KERNELS ? [
    (AK.MergeSort(), (Int32, UInt32, Float32), 1000),
    (AK.RadixSort(), (UInt32, Int32, Float32, UInt64, Int64, Float64), 200),
    (AK.BitonicSort(), (UInt8, Int16, Int32, UInt32, Float32, Int64, UInt64, Float64), 20),
] : [(SORT_ALG, (Int32, UInt32, Float32), 1000)]

# The permutation algorithms (`Auto()` always picks the merge sort on GPUs)
PERM_ALGS = TEST_KERNELS ? [SORT_ALG, AK.MergeSort(lowmem=true)] : [SORT_ALG]

@testset "sort: random inputs" begin
    Random.seed!(0)
    for (alg, eltypes, n) in SORT_FUZZ, T in valid_backend_eltypes(BACKEND, eltypes), _ in 1:n
        vh = rand(T, rand(1:100_000))
        v = array_from_host(vh)
        AK.sort!(v; alg)
        @test Array(v) == sort(vh)
    end
end


@testset "sort: settings" begin
    Random.seed!(0)
    alg = sort_alg(block_size=64, max_tasks=64, min_elems=8)
    v = array_from_host(rand(1:100_000, 10_000), Float32)
    with_workspace(AK.sort!, v; alg, lt=(>), by=abs, rev=true)
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Int32)
    with_workspace(AK.sort!, v; alg, lt=(>), rev=true)
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Float32)
    v = with_workspace(AK.sort, v; alg, lt=(>), by=abs, rev=true)
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Int32)
    v = with_workspace(AK.sort, v; alg, lt=(>), by=abs, rev=true)
    @test issorted(Array(v))
end


@testset "sort: by" begin
    # `by` keys, including type-changing ones, give Base's order, for the stable algorithms
    Random.seed!(42)
    alg = TEST_KERNELS ? AK.MergeSort() : AK.CPUThreads.SampleSort()
    for T in valid_backend_eltypes(BACKEND, (Float32, Float64, Int32))
        vh = T <: AbstractFloat ? randn(T, 10_000) : rand(T(-100):T(100), 10_000)
        for kw in ((by=abs,), (by=abs, rev=true), (by=x -> x^2,))
            @test Array(AK.sort!(array_from_host(vh); alg, kw...)) == sort(vh; kw...)
        end
    end
    vh = randn(Float32, 10_000)
    for kw in ((by=x -> x > 0,), (rev=true,), (;))
        @test Array(AK.sort!(array_from_host(vh); alg, kw...)) == sort(vh; kw...)
    end
    for n in (1, 2, 513, 1025)
        vh = randn(Float32, n)
        @test Array(AK.sort!(array_from_host(vh); alg, by=abs)) == sort(vh; by=abs)
    end
    vh = randn(Float32, 20_000)
    @test Array(with_workspace(AK.sort!, array_from_host(vh); alg, by=abs)) == sort(vh; by=abs)
    @test Array(AK.sort!(array_from_host(vh); by=abs)) == sort(vh; by=abs)

    # Keys keep their own type: transforms to wider floats, fractions and tuples must not be
    # converted back to the element type
    if KernelAbstractions.supports_float64(BACKEND)     # the keys are Float64
        vh = Float32[2, 1, 3]
        by_wide = x -> 1.0 + Float64(x) * eps(Float64)
        @test Array(AK.sort!(array_from_host(vh); alg, by=by_wide)) == sort(vh; by=by_wide)
        vh = Int32[3, 1, 2, 4]
        @test Array(AK.sort!(array_from_host(vh); alg, by=x -> x / 2)) == sort(vh; by=x -> x / 2)
    end
    vh = Int32[3, 1, 2, 4]
    @test Array(AK.sort!(array_from_host(vh); alg, by=x -> (x % 2, x))) ==
          sort(vh; by=x -> (x % 2, x))
end


@testset "sort: rejected algorithms" begin
    # The entry points reject what resolution does (see test/algorithms.jl), before sorting
    vh = rand(Int32, 128)
    v = array_from_host(vh)
    ix = array_from_host(zeros(Int, length(vh)))
    if TEST_KERNELS
        @test_throws ArgumentError AK.sortperm!(ix, v; alg=AK.RadixSort())
        @test_throws ArgumentError AK.sortperm!(ix, v; alg=AK.BitonicSort())
        @test_throws ArgumentError AK.sortperm(v; alg=AK.BitonicSort())
        @test_throws ArgumentError AK.sort!(v; alg=AK.MergeSort(lowmem=true))
        @test_throws ArgumentError AK.sort!(v; alg=AK.RadixSort(), by=abs)
        @test_throws ArgumentError AK.sort!(v; alg=AK.RadixSort(), lt=(>))
        @test_throws ArgumentError AK.sort!(v; alg=AK.RadixSort(),
                                            order=Base.Order.By(abs, Base.Order.Forward))
        @test_throws ArgumentError AK.sort!(array_from_host(rand(Int16, 10)); alg=AK.RadixSort())
        @test_throws ArgumentError AK.sort!(v; alg=AK.BitonicSort(block_size=100))
        @test_throws ArgumentError AK.sort!(v; alg=AK.BitonicSort(items_per_thread=3))
        @test_throws ArgumentError AK.sort!(v; alg=AK.BitonicSort(
            block_size=2, items_per_thread=1 << (Sys.WORD_SIZE - 2)))
    end
    if AK.runs_threads(BACKEND)
        @test_throws ArgumentError AK.sort!(v; alg=AK.CPUThreads.SampleSort(max_tasks=0))
        @test_throws ArgumentError AK.sortperm!(ix, v; alg=AK.RadixSort())
    else
        @test_throws ArgumentError AK.sort!(v; alg=AK.CPUThreads.SampleSort())
    end
    # AK's kernels need a backend that runs them (on the host, KernelAbstractions 0.10)
    if !AK.runs_kernels(BACKEND)
        for alg in (AK.MergeSort(), AK.RadixSort(), AK.BitonicSort())
            @test_throws ArgumentError AK.sort!(v; alg)
        end
    end
    @test Array(v) == vh
end


if AK.runs_threads(BACKEND)
@testset "sort: SampleSort stability" begin
    # Stable, also when several tasks sort buckets in parallel (at least 16 elements per task):
    # elements tagged with their position, sorted by a key with many ties, keep their order
    Random.seed!(0)
    th = [(rand(Int32(1):Int32(50)), Int32(i)) for i in 1:100_000]
    for max_tasks in (1, 4, 16)
        t = copy(th)
        AK.sort!(t; by=first, alg=AK.CPUThreads.SampleSort(; max_tasks))
        @test t == sort(th; by=first)
        ix = zeros(Int, length(th))
        AK.sortperm!(ix, th; by=first, alg=AK.CPUThreads.SampleSort(; max_tasks))
        @test ix == sortperm(th; by=first)
    end
end
end


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


if TEST_KERNELS
@testset "sort: RadixSort" begin
    Random.seed!(0)
    alg = AK.RadixSort()

    for T in valid_backend_eltypes(BACKEND, (UInt32, Int32, Float32, UInt64, Int64, Float64))
        v_h = rand(T, 10_000)
        v = array_from_host(v_h)
        AK.sort!(v; alg, rev=true)
        @test Array(v) == sort(v_h; rev=true)
    end

    # Floating-point ordering
    for T in valid_backend_eltypes(BACKEND, (Float32, Float64))
        specials = T[1, -0.0, 0.0, NaN, -NaN, Inf, -Inf, 2.5, -2.5,
                     prevfloat(zero(T)), nextfloat(zero(T))]
        v_h = shuffle!(vcat(specials, randn(T, 10_000)))
        for rev in (false, true)
            v = array_from_host(v_h)
            AK.sort!(v; alg, rev)
            @test isequal(Array(v), sort(v_h; rev))
        end
    end

    # Ordering composition
    v_h = rand(Int32, 10_000)
    for (rev, order) in ((nothing, Base.Order.Reverse), (true, Base.Order.Forward),
                         (true, Base.Order.Reverse))
        v = array_from_host(v_h)
        AK.sort!(v; alg, rev, order)
        @test Array(v) == sort(v_h; rev, order)
    end

    @test isempty(Array(AK.sort!(array_from_host(Int32[]); alg)))
    @test Array(AK.sort!(array_from_host(Int32[42]); alg)) == Int32[42]
    @test Array(AK.sort!(array_from_host(Int32[2, 1]); alg)) == Int32[1, 2]

    # A workspace, and out of place, leaving the input unchanged
    v_h = rand(Float32, 50_000)
    v = array_from_host(v_h)
    with_workspace(AK.sort!, v; alg)
    @test Array(v) == sort(v_h)
    v_h = rand(Float32, 10_000)
    v = array_from_host(v_h)
    w = AK.sort(v; alg)
    @test Array(w) == sort(v_h)
    @test Array(v) == v_h

    # Tuning parameters and single-block boundaries
    v_h = rand(UInt32, 20_000)
    for block_size in (128, 256, 512), items_per_thread in (1, 2, 4)
        v = array_from_host(v_h)
        AK.sort!(v; alg=AK.RadixSort(; block_size, items_per_thread))
        @test Array(v) == sort(v_h)
    end
    for n in (255, 256, 257)
        v_h = rand(UInt32, n)
        v = array_from_host(v_h)
        AK.sort!(v; alg=AK.RadixSort(block_size=128))
        @test Array(v) == sort(v_h)
    end

    # Chunk sizes, in the passes and in the single-block sort (small chunks need small blocks for
    # their histograms to fit local memory); a chunk larger than the block makes the portable
    # kernels run instead
    for (block_size, chunk_size) in ((8, 1), (32, 8), (256, 64), (256, 256), (256, 512)),
        n in (12, 300, 20_000)
        v_h = rand(Int32(0):Int32(9), n)
        v = array_from_host(v_h)
        AK.sort!(v; alg=AK.RadixSort(; block_size, chunk_size))
        @test Array(v) == sort(v_h)
    end
end
end


if TEST_KERNELS
@testset "sort: BitonicSort" begin
    Random.seed!(0)
    alg = AK.BitonicSort()

    # Lengths around the tile size (2048 by default) and through several global levels
    for n in (1, 2, 3, 7, 8, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096,
              4097, 8191, 8192, 8193, 100_000, 1_000_000)
        v_h = rand(Float32, n)
        v = array_from_host(v_h)
        AK.sort!(v; alg)
        @test Array(v) == sort(v_h)
    end

    # Adversarial patterns
    for n in (1000, 8192, 65536)
        for v_h in (fill(2.5f0, n), Float32.(1:n), Float32.(n:-1:1),
                    Float32.(rand(0:1, n)), Float32.(rand(0:3, n)))
            v = array_from_host(v_h)
            AK.sort!(v; alg)
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
        AK.sort!(v; alg, rev)
        @test isequal(Array(v), sort(v_h; rev))
    end

    # lt, by, rev and order; the sort is not stable, so only the order of the keys is Base's
    v_h = rand(Int32, 10_000)
    for kw in ((rev=true,), (order=Base.Order.Reverse,), (rev=true, order=Base.Order.Reverse),
               (lt=(>),), (by=abs,), (by=x -> x % Int32(7), rev=true),
               (lt=(a, b) -> a % 5 < b % 5,))
        v = array_from_host(v_h)
        AK.sort!(v; alg, kw...)
        sorted = Array(v)
        ord = Base.Order.ord(get(kw, :lt, isless), get(kw, :by, identity),
                             get(kw, :rev, nothing), get(kw, :order, Base.Order.Forward))
        @test issorted(sorted; order=ord)
        @test sort(sorted) == sort(v_h)
    end

    # Tuning
    v_h = rand(UInt32, 20_000)
    for block_size in (32, 64, 128, 256, 512), items_per_thread in (1, 2, 8, 16)
        v = array_from_host(v_h)
        AK.sort!(v; alg=AK.BitonicSort(; block_size, items_per_thread))
        @test Array(v) == sort(v_h)
    end

    @test isempty(Array(AK.sort!(array_from_host(Int32[]); alg)))

    # Matrices are sorted as one flat vector by default
    m_h = rand(Float32, 100, 30)
    m = array_from_host(m_h)
    AK.sort!(m; alg)
    @test vec(Array(m)) == sort(vec(m_h))

    # Out of place, leaving the input unchanged
    v_h = rand(Float32, 10_000)
    v = array_from_host(v_h)
    w = AK.sort(v; alg)
    @test Array(w) == sort(v_h)
    @test Array(v) == v_h
end
end


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


@testset "sort: backend-free inputs" begin
    # A range has no backend: the result is allocated on the one given
    r = AK.sort(5:-1:1; backend=BACKEND)
    @test get_backend(r) == BACKEND
    @test Array(r) == 1:5
    ix = AK.sortperm(5:-1:1; backend=BACKEND)
    @test get_backend(ix) == BACKEND
    @test Array(ix) == 5:-1:1
end


if TEST_KERNELS
@testset "sort: reshaped views" begin
    # Merge sort's kernels read such an input through `@Const`, which rebuilds the reshape on the
    # device
    h = rand(Int32, 50, 40)
    d = array_from_host(h)
    v = vec(view(d, 1:40, 1:30))
    AK.sort!(v; alg=AK.MergeSort())
    @test Array(v) == sort(vec(view(h, 1:40, 1:30)))
end
end


@testset "sort: offset axes" begin
    # Arrays whose axes do not start at 1; permutations hold their indices, as Base's do
    Random.seed!(0)
    threads = AK.CPUThreads.SampleSort(max_tasks=4, min_elems=8)
    sort_algs = TEST_KERNELS ?
        [SORT_ALG, AK.MergeSort(block_size=64), AK.RadixSort(), AK.BitonicSort()] :
        [SORT_ALG, threads]
    perm_algs = TEST_KERNELS ?
        [SORT_ALG, AK.MergeSort(block_size=64), AK.MergeSort(block_size=64, lowmem=true)] :
        [SORT_ALG, threads]
    pair_algs = TEST_KERNELS ? [SORT_ALG, AK.MergeSort(block_size=64)] : [SORT_ALG, threads]

    h = rand(Int32, 1000)
    hm = rand(Int32(1):Int32(50), 30, 40)
    for alg in sort_algs
        v = offset_from_host(h, (-3,))
        AK.sort!(v; alg)
        @test Array(v.parent) == sort(h)
        with_workspace(AK.sort!, v; rev=true, alg)
        @test Array(v.parent) == sort(h; rev=true)
        r = AK.sort(offset_from_host(h, (-3,)); alg)
        @test axes(r) == axes(v) && Array(r.parent) == sort(h)
        v = offset_from_host(Int32[5], (-3,))
        AK.sort!(v; alg)
        @test Array(v.parent) == [5]
        for dims in (1, 2)
            m = offset_from_host(hm, (-1, 3))
            AK.sort!(m; dims, alg)
            @test Array(m.parent) == sort(hm; dims)
        end
    end

    for alg in perm_algs
        v = offset_from_host(h, (-3,))
        for ix in (offset_from_host(zeros(Int, 1000), (7,)), array_from_host(zeros(Int, 1000)))
            AK.sortperm!(ix, v; alg)
            @test Array(ix isa OffsetAxes ? ix.parent : ix) == sortperm(h) .- 3
        end
        ix = offset_from_host(zeros(Int, 1000), (-3,))
        with_workspace(AK.sortperm!, ix, v; dims=1, rev=true, alg)
        @test Array(ix.parent) == sortperm(h; rev=true) .- 3
        @test Array(AK.sortperm(v; alg)) == sortperm(h) .- 3
        @test Array(AK.sortperm(offset_from_host(Int32[5], (-3,)); alg)) == [-2]
        for dims in (1, 2)
            m = offset_from_host(hm, (-1, 3))
            ix = offset_from_host(zeros(Int, 30, 40), (-1, 3))
            AK.sortperm!(ix, m; dims, alg)
            @test Array(ix.parent) == sortperm(hm; dims)
            @test Array(AK.sortperm(m; dims, alg)) == sortperm(hm; dims)
        end
    end

    for alg in pair_algs
        k = rand(Int32(1):Int32(50), 1000)
        keys, values = offset_from_host(k, (-3,)), offset_from_host(Int32.(1:1000), (4,))
        AK.sort_by_key!(keys, values; alg)
        @test Array(keys.parent) == sort(k) && Array(values.parent) == sortperm(k)
        for dims in (1, 2)
            hv = rand(Int32, 30, 40)
            keys, values = offset_from_host(hm, (-1, 3)), offset_from_host(hv, (-1, 3))
            AK.sort_by_key!(keys, values; dims, alg)
            @test Array(keys.parent) == sort(hm; dims) &&
                  Array(values.parent) == hv[sortperm(hm; dims)]
        end
    end
end


if TEST_KERNELS && TEST_BITS_UNIONS
@testset "sort: bits-union elements" begin
    # Merge sort's kernels read such elements through `@Const` (where the backend's arrays can
    # hold them)
    Random.seed!(0)
    h = rand(Union{Missing, Int32}[missing, 1, 2, 3], 3000)
    @test isequal(Array(AK.sort!(array_from_host(h); alg=AK.MergeSort())), sort(h))
    k = rand(Int32(1):Int32(50), 3000)
    kd, vd = array_from_host(k), array_from_host(h)
    AK.sort_by_key!(kd, vd; alg=AK.MergeSort())
    @test Array(kd) == sort(k) && isequal(Array(vd), h[sortperm(k)])
end
end
