# The sorting algorithms of the configuration, with the element types and the number of random
# inputs each sorts. `Auto()` picks among them by the device's tuning, and on the host always the
# sample sort.
SORT_FUZZ = TEST_KERNELS ? [
    (AK.MergeSort(), (Int32, UInt32, Float32), 1000),
    (AK.RadixSort(), (UInt32, Int32, Float32, UInt64, Int64, Float64), 200),
    (AK.BitonicSort(), (UInt8, Int16, Int32, UInt32, Float32, Int64, UInt64, Float64), 20),
] : [(SORT_ALG, (Int32, UInt32, Float32), 1000)]


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
        for alg in (AK.MergeSort(block_size=3), AK.MergeSort(block_size=96))
            @test_throws ArgumentError AK.sort!(v; alg)
            @test_throws ArgumentError AK.sortperm!(ix, v; alg)
            @test_throws ArgumentError AK.sort_by_key!(v, copy(ix); alg)
        end
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


# 256 bytes, ordered by `key`
struct Wide
    key::Int64
    pad::NTuple{31, Int64}
end
Wide(key) = Wide(key, ntuple(_ -> 0, 31))
Base.isless(a::Wide, b::Wide) = isless(a.key, b.key)

if TEST_KERNELS
@testset "sort: large elements" begin
    # Tiles of the default sizes would need more local memory than devices have: merge sort
    # shrinks its default block, bitonic sort rejects them. (`array_from_host` zero-fills, which
    # OpenCL.jl cannot do for elements of more than 128 bytes.)
    Random.seed!(0)
    on_device(h) = copyto!(KernelAbstractions.allocate(BACKEND, eltype(h), size(h)), h)
    h = [Wide(rand(1:100)) for _ in 1:3000]
    @test Array(AK.sort!(on_device(h); alg=AK.MergeSort())) == sort(h)
    @test Array(AK.sortperm(on_device(h); alg=AK.MergeSort())) == sortperm(h)
    k = rand(Int32(1):Int32(50), 3000)
    kd, vd = array_from_host(k), on_device(h)
    AK.sort_by_key!(kd, vd; alg=AK.MergeSort())
    @test Array(kd) == sort(k) && Array(vd) == h[sortperm(k)]
    @test_throws ArgumentError AK.sort!(on_device(h); alg=AK.BitonicSort())
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
