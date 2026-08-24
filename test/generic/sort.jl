if !prefer_threads
@testset "merge_sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    # Testing different settings
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort!(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    AK.merge_sort!(v, lt=(>), rev=true,
                block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Float32)
    v = AK.merge_sort(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(1:10_000, Int32)
    v = AK.merge_sort(v, lt=(>), by=abs, rev=true,
                block_size=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))
end

@testset "sort_by_transform" begin
    # Tests for the by= hoisting optimisation: by(elem) is broadcast once before
    # sorting rather than being called inside every merge comparison.
    # Checks exact output match against Base.sort so we catch ordering regressions.
    Random.seed!(42)

    # Exact match against Base.sort for common by= functions
    for T in valid_backend_eltypes(BACKEND, (Float32, Float64, Int32))
        n   = 10_000
        v_h = T <: AbstractFloat ? randn(T, n) : rand(T(-100):T(100), n)
        for (kw, base_kw) in (
            ((by=abs,),                (by=abs,)),
            ((by=abs, rev=true),       (by=abs, rev=true)),
            ((by=x->x^2,),             (by=x->x^2,)),
        )
            v   = array_from_host(v_h)
            tmp = copy(v)
            AK.merge_sort!(tmp; kw...)
            @test Array(tmp) == sort(v_h; base_kw...)
        end
    end

    # rev=true and lt=(>) are not hoisted (no by=) — verify they still pass
    n   = 10_000
    v_h = randn(Float32, n)
    v   = array_from_host(v_h); tmp = copy(v)
    AK.merge_sort!(tmp; rev=true)
    @test Array(tmp) == sort(v_h; rev=true)

    # Edge sizes under by= hoisting
    for n in (1, 2, 513, 1025)
        v_h = randn(Float32, n)
        v   = array_from_host(v_h)
        tmp = copy(v)
        AK.merge_sort!(tmp; by=abs)
        @test Array(tmp) == sort(v_h; by=abs)
    end

    # temp kwarg still forwarded correctly through hoisting path
    n    = 20_000
    v_h  = randn(Float32, n)
    v    = array_from_host(v_h)
    tmp  = copy(v)
    temp = array_from_host(zeros(Float32, n))
    AK.merge_sort!(tmp; by=abs, temp)
    @test Array(tmp) == sort(v_h; by=abs)

    # sort! (public API) routes through the same hoisting path
    n   = 10_000
    v_h = randn(Float32, n)
    v   = array_from_host(v_h)
    tmp = copy(v)
    AK.sort!(tmp; by=abs)
    @test Array(tmp) == sort(v_h; by=abs)

    # by= with a type-changing transform (Float32 → Bool key)
    n   = 10_000
    v_h = randn(Float32, n)
    v   = array_from_host(v_h)
    tmp = copy(v)
    AK.merge_sort!(tmp; by=x->x>0)
    @test Array(tmp) == sort(v_h; by=x->x>0)

    # identity path unchanged: verify no regression from the early-return guard
    n   = 10_000
    v_h = rand(Float32, n)
    v   = array_from_host(v_h)
    tmp = copy(v)
    AK.merge_sort!(tmp)
    @test Array(tmp) == sort(v_h)
end

else # CPU backend
@testset "sample_sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        AK.sample_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        AK.sample_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        AK.sample_sort!(v)
        vh = Array(v)
        @test issorted(vh)
    end

    # Testing different settings
    v = array_from_host(rand(1:100_000, 10_000), Float32)
    AK.sample_sort!(v, lt=(>), by=abs, rev=true,
                    max_tasks=64, temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Int32)
    AK.sample_sort!(v, lt=(>), rev=true,
                    max_tasks=64, temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))
end
end


@testset "sort" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Int32, num_elems))
        AK.sort!(v; prefer_threads)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(UInt32, num_elems))
        AK.sort!(v; prefer_threads)
        vh = Array(v)
        @test issorted(vh)
    end

    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        AK.sort!(v; prefer_threads)
        vh = Array(v)
        @test issorted(vh)
    end

    # Testing different settings
    v = array_from_host(rand(1:100_000, 10_000), Float32)
    AK.sort!(v; prefer_threads, lt=(>), by=abs, rev=true,
            max_tasks=64, min_elems=8, block_size=64,
            temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Int32)
    AK.sort!(v; prefer_threads, lt=(>), rev=true,
            max_tasks=64, min_elems=8, block_size=64,
            temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Float32)
    v = AK.sort(v; prefer_threads, lt=(>), by=abs, rev=true,
                max_tasks=64, min_elems=8, block_size=64,
                temp=array_from_host(1:10_000, Float32))
    @test issorted(Array(v))

    v = array_from_host(rand(1:100_000, 10_000), Int32)
    v = AK.sort(v; prefer_threads, lt=(>), by=abs, rev=true,
                max_tasks=64, min_elems=8, block_size=64,
                temp=array_from_host(1:10_000, Int32))
    @test issorted(Array(v))
end


@testset "sort_alg_kwarg" begin
    Random.seed!(2026)

    function is_valid_perm(vh, ixh; kwargs...)
        n = length(vh)
        length(ixh) == n &&
        sort(Int.(ixh)) == collect(1:n) &&
        issorted(vh[ixh]; kwargs...)
    end

    if !prefer_threads
        for T in valid_backend_eltypes(BACKEND,
                        (UInt32, Int32, Float32, UInt64, Int64, Float64))
            v_h = rand(T, 10_000)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort())
            @test Array(v) == sort(v_h)
        end

        v_h = rand(Int32, 10_000)
        v_default = array_from_host(v_h)
        v_merge = array_from_host(v_h)
        AK.sort!(v_default; prefer_threads)
        AK.sort!(v_merge; prefer_threads, alg=AK.MergeSort())
        @test Array(v_merge) == Array(v_default)

        perm_h = rand(Float32, 4096)
        for alg in (AK.MergeSort(), AK.MergeSort(lowmem=true))
            v = array_from_host(perm_h)
            ix = array_from_host(zeros(Int, length(perm_h)))
            temp = array_from_host(zeros(Int, length(perm_h)))
            AK.sortperm!(ix, v; prefer_threads, alg, temp)
            @test is_valid_perm(perm_h, Int.(Array(ix)))
        end

        v = array_from_host(rand(Float32, 128))
        ix = array_from_host(zeros(Int, length(v)))
        @test_throws ArgumentError AK.sort!(copy(v); prefer_threads, alg=AK.SampleSort())
        @test_throws ArgumentError AK.sortperm!(ix, v; prefer_threads, alg=AK.RadixSort())
    else
        v_h = rand(Int32, 10_000)
        v_default = array_from_host(v_h)
        v_sample = array_from_host(v_h)
        AK.sort!(v_default; prefer_threads)
        AK.sort!(v_sample; prefer_threads, alg=AK.SampleSort())
        @test Array(v_sample) == Array(v_default)

        ix = array_from_host(zeros(Int, length(v_h)))
        AK.sortperm!(ix, array_from_host(v_h); prefer_threads, alg=AK.SampleSort())
        @test is_valid_perm(v_h, Int.(Array(ix)))

        @test_throws ArgumentError AK.sort!(array_from_host(v_h); prefer_threads, alg=AK.MergeSort())
        @test_throws ArgumentError AK.sort!(array_from_host(v_h); prefer_threads, alg=AK.RadixSort())
        @test_throws ArgumentError AK.sortperm!(ix, array_from_host(v_h); prefer_threads, alg=AK.RadixSort())
    end
end


if !prefer_threads
@testset "merge_sort_by_key" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Int32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(UInt32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        k = array_from_host(rand(Float32, num_elems))
        v = copy(k) .- 1
        AK.merge_sort_by_key!(k, v)
        kh = Array(k)
        vh = Array(v)
        @test issorted(kh)
        @test issorted(vh)
    end

    # Testing different settings
    k = array_from_host(1:10_000, Float32)
    v = array_from_host(1:10_000, Int32)
    AK.merge_sort_by_key!(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Float32),
                        temp_values=array_from_host(1:10_000, Int32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort_by_key!(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Int32),
                        temp_values=array_from_host(1:10_000, Float32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Float32)
    v = array_from_host(1:10_000, Int32)
    AK.merge_sort_by_key(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Float32),
                        temp_values=array_from_host(1:10_000, Int32))
    @test issorted(Array(k))
    @test issorted(Array(v))

    k = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sort_by_key(k, v,
                        lt=(>), by=abs, rev=true,
                        block_size=64,
                        temp_keys=array_from_host(1:10_000, Int32),
                        temp_values=array_from_host(1:10_000, Float32))
    @test issorted(Array(k))
    @test issorted(Array(v))
end
end


if !prefer_threads
@testset "merge_sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sortperm!(ix,
                    v,
                    lt=(>), by=abs, rev=true,
                    inplace=true, block_size=64,
                    temp_ix=array_from_host(1:10_000, Int32),
                    temp_v=array_from_host(1:10_000, Float32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.merge_sortperm(v,
                        lt=(>), by=abs, rev=true,
                        inplace=true, block_size=64,
                        temp_ix=array_from_host(1:10_000, Int),
                        temp_v=array_from_host(1:10_000, Float32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end

else # CPU backend
    @testset "sample_sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.sample_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.sample_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.sample_sortperm!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.sample_sortperm!(ix,
                    v,
                    lt=(>), by=abs, rev=true,
                    max_tasks=64,
                    temp=array_from_host(1:10_000, Int32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end
end


if !prefer_threads
@testset "merge_sortperm_lowmem" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.merge_sortperm_lowmem!(ix, v)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.merge_sortperm_lowmem!(ix,
                            v,
                            lt=(>), by=abs, rev=true,
                            block_size=64,
                            temp=array_from_host(1:10_000, Int32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.merge_sortperm_lowmem(v,
                                lt=(>), by=abs, rev=true,
                                block_size=64,
                                temp=array_from_host(1:10_000, Int))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end
end


@testset "sortperm" begin
    Random.seed!(0)

    # Fuzzy correctness testing
    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Int32, num_elems))
        AK.sortperm!(ix, v; prefer_threads)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(UInt32, num_elems))
        AK.sortperm!(ix, v; prefer_threads)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    for _ in 1:1000
        num_elems = rand(1:100_000)
        ix = array_from_host(zeros(Int32, num_elems))
        v = array_from_host(rand(Float32, num_elems))
        AK.sortperm!(ix, v; prefer_threads)
        ixh = Array(ix)
        vh = Array(v)
        @test issorted(vh[ixh])
    end

    # Testing different settings
    ix = array_from_host(1:10_000, Int32)
    v = array_from_host(1:10_000, Float32)
    AK.sortperm!(ix,
                v;
                prefer_threads,
                lt=(>), by=abs, rev=true,
                block_size=64,
                temp=array_from_host(1:10_000, Int32))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])

    v = array_from_host(1:10_000, Float32)
    ix = AK.sortperm(v;
                    prefer_threads,
                    lt=(>), by=abs, rev=true,
                    block_size=64,
                    temp=array_from_host(1:10_000, Int))
    ixh = Array(ix)
    vh = Array(v)
    @test issorted(vh[ixh])
end


if !prefer_threads
@testset "sortperm_extended" begin
    # Helper: ix is a valid permutation of 1:n that produces a sorted order
    function is_valid_perm(vh, ixh; kwargs...)
        n = length(vh)
        length(ixh) == n &&
        sort(Int.(ixh)) == collect(1:n) &&
        issorted(vh[ixh]; kwargs...)
    end

    # ── Element types ────────────────────────────────────────────────────────
    Random.seed!(123)

    for T in valid_backend_eltypes(BACKEND, (Int16, UInt16, Int64, UInt64, Float64, UInt8))
        for _ in 1:50
            n  = rand(1:50_000)
            v  = array_from_host(rand(T, n))
            ix = array_from_host(zeros(Int, n))
            AK.sortperm!(ix, v)
            vh, ixh = Array(v), Array(ix)
            @test is_valid_perm(vh, ixh)
        end
    end

    # ── Edge sizes ───────────────────────────────────────────────────────────
    for n in (1, 2, 3, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049)
        v  = array_from_host(rand(Float32, n))
        ix = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, v)
        vh, ixh = Array(v), Array(ix)
        @test is_valid_perm(vh, ixh)
    end

    # ── Data distributions ───────────────────────────────────────────────────
    n = 2^14
    Random.seed!(456)
    base = rand(Float32, n)

    for arr in (
        sort(base),                                # already sorted
        reverse(sort(base)),                       # reverse sorted
        fill(1f0, n),                              # all same
        Float32.(rand(1:4, n)),                    # 4 unique values
    )
        v  = array_from_host(arr)
        ix = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, v)
        vh, ixh = Array(v), Array(ix)
        @test is_valid_perm(vh, ixh)
    end

    # ── Comparator options ───────────────────────────────────────────────────
    n = 10_000
    Random.seed!(789)

    for kw in (
        (rev=true,),
        (by=abs,),
        (by=abs, rev=true),
        (lt=(>),)
    )
        v  = array_from_host(randn(Float32, n))
        ix = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, v; kw...)
        vh, ixh = Array(v), Array(ix)
        res = is_valid_perm(vh, ixh; kw...)
        @test res
    end

    # ── temp kwarg: buffer reuse gives identical result ───────────────────────
    n = 20_000
    Random.seed!(321)
    v1   = array_from_host(rand(Float32, n))
    v2   = copy(v1)
    ix1  = array_from_host(zeros(Int, n))
    ix2  = array_from_host(zeros(Int, n))
    temp = array_from_host(zeros(Int, n))
    AK.sortperm!(ix1, v1; temp)
    AK.sortperm!(ix2, v2; temp)
    @test Array(ix1) == Array(ix2)

    # ── Exact match against Base.sortperm ────────────────────────────────────
    for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
        n   = 10_000
        v_h = rand(T, n)
        ref = sortperm(v_h)
        v   = array_from_host(v_h)
        ix  = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, v)
        ixh = Int.(Array(ix))
        @test v_h[ixh] == v_h[ref]
    end

    # ── Stability: equal keys must preserve original relative order ───────────
    n   = 10_000
    v_h = Int32.(mod.(1:n, 10))   # values 0..9 cycling, 1000 of each
    v   = array_from_host(v_h)
    ix  = array_from_host(zeros(Int, n))
    AK.sortperm!(ix, v)
    ixh = Array(ix)
    for k in 0:9
        group = ixh[v_h[ixh] .== k]
        @test issorted(group)   # within each equal-key group, indices must be ascending
    end

    # ── sortperm does not mutate the input ───────────────────────────────────
    v    = array_from_host(rand(Float32, 5_000))
    vbak = copy(v)
    AK.sortperm(v)
    @test Array(v) == Array(vbak)
end

@testset "radix_sort_alg" begin
    if !prefer_threads
        Random.seed!(0)

        # ── Correctness: fuzzy testing across supported types ─────────────────
        for T in (UInt32, Int32, Float32)
            for _ in 1:200
                n = rand(1:100_000)
                v = array_from_host(rand(T, n))
                AK.sort!(v; prefer_threads, alg=AK.RadixSort())
                @test issorted(Array(v))
            end
        end

        for T in valid_backend_eltypes(BACKEND,
                        (UInt64, Int64, Float64))
            for _ in 1:200
                n = rand(1:100_000)
                v = array_from_host(rand(T, n))
                AK.sort!(v; prefer_threads, alg=AK.RadixSort())
                @test issorted(Array(v))
            end
        end

        # ── Exact match against Base.sort ─────────────────────────────────────
        for T in valid_backend_eltypes(BACKEND,
                        (UInt32, Int32, Float32, UInt64, Int64, Float64))
            n   = 10_000
            v_h = rand(T, n)
            v   = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort())
            @test Array(v) == sort(v_h)
        end

        # ── rev=true ──────────────────────────────────────────────────────────
        for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND),
                        (UInt32, Int32, Float32, UInt64, Int64, Float64))
            n   = 10_000
            v_h = rand(T, n)
            v   = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort(), rev=true)
            @test Array(v) == sort(v_h; rev=true)
        end

        # Floating-point ordering
        for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND),
                        (Float32, Float64))
            specials = T[1, -0.0, 0.0, NaN, -NaN, Inf, -Inf, 2.5, -2.5,
                         prevfloat(zero(T)), nextfloat(zero(T))]
            v_h = shuffle!(vcat(specials, randn(T, 10_000)))

            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort())
            @test isequal(Array(v), sort(v_h))

            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort(), rev=true)
            @test isequal(Array(v), sort(v_h; rev=true))
        end

        # Ordering composition
        v_h = rand(Int32, 10_000)
        for (rev, order) in ((nothing, Base.Order.Reverse), (true, Base.Order.Forward),
                             (true, Base.Order.Reverse))
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort(), rev, order)
            @test Array(v) == sort(v_h; rev, order)
        end

        # ── Stability: equal keys → result matches sort (radix is stable) ───
        n   = 10_000
        v_h = Int32.(mod.(1:n, 100))   # 100 distinct values, 100 copies each
        v   = array_from_host(v_h)
        AK.sort!(v; prefer_threads, alg=AK.RadixSort())
        @test Array(v) == sort(v_h)

        # ── Edge cases ────────────────────────────────────────────────────────
        @test length(Array(AK.sort!(array_from_host(Int32[]); prefer_threads, alg=AK.RadixSort()))) == 0
        @test Array(AK.sort!(array_from_host(Int32[42]); prefer_threads, alg=AK.RadixSort())) == Int32[42]
        @test Array(AK.sort!(array_from_host(Int32[2, 1]); prefer_threads, alg=AK.RadixSort())) == Int32[1, 2]

        # ── temp kwarg: preallocated buffer ───────────────────────────────────
        n    = 50_000
        v_h  = rand(Float32, n)
        v    = array_from_host(v_h)
        temp = similar(v)
        AK.sort!(v; prefer_threads, alg=AK.RadixSort(), temp)
        @test Array(v) == sort(v_h)

        # ── Out-of-place ──────────────────────────────────────────────────────
        n   = 10_000
        v_h = rand(Float32, n)
        v   = array_from_host(v_h)
        w   = AK.sort(v; prefer_threads, alg=AK.RadixSort())
        @test Array(w) == sort(v_h)
        @test Array(v) == v_h   # input unchanged

        # Tuning parameters and single-block boundaries
        v_h = rand(UInt32, 20_000)
        for block_size in (128, 256, 512), items_per_thread in (1, 2, 4)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads,
                     alg=AK.RadixSort(; block_size, items_per_thread))
            @test Array(v) == sort(v_h)
        end

        for n in (255, 256, 257)
            v_h = rand(UInt32, n)
            v = array_from_host(v_h)
            AK.sort!(v; prefer_threads, alg=AK.RadixSort(block_size=128))
            @test Array(v) == sort(v_h)
        end

        # ── Rejected: custom by/lt, unsupported element type ────────────────────
        n   = 10
        v_h = rand(Int32, n)
        v   = array_from_host(v_h)
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.RadixSort(), by=abs)

        v_h = rand(Int32, n)
        v   = array_from_host(v_h)
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.RadixSort(), lt=(>))

        v_h = rand(Int32, n)
        v = array_from_host(v_h)
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.RadixSort(),
                                             order=Base.Order.By(abs, Base.Order.Forward))

        v_h = rand(Int16, n)
        v   = array_from_host(v_h)
        @test_throws ArgumentError AK.sort!(v; prefer_threads, alg=AK.RadixSort())
    end
end


@testset "sort_dims" begin
    Random.seed!(0)

    # Fuzzy correctness against Base.sort(A; dims) for 2D and 3D arrays
    for _ in 1:200
        nd = rand(2:3)
        sz = ntuple(_ -> rand(1:15), nd)
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32))
            A_h = rand(T, sz...)
            A   = array_from_host(A_h)
            for dim in 1:nd, rev in (false, true)
                @test Array(AK.sort(A; prefer_threads, dims=dim, rev=rev)) ==
                      sort(A_h; dims=dim, rev=rev)
            end
        end
    end

    # by= and order= act on the values within each slice
    A_h = rand(Float32, 17, 23)
    A   = array_from_host(A_h)
    @test Array(AK.sort(A; prefer_threads, dims=1, by=x->-x)) == sort(A_h; dims=1, by=x->-x)
    @test Array(AK.sort(A; prefer_threads, dims=2,
                        order=Base.Order.Reverse)) == sort(A_h; dims=2, order=Base.Order.Reverse)

    # In-place sorts each slice, leaves the array otherwise intact
    A_h = rand(Int32, 40, 31)
    A   = array_from_host(A_h)
    AK.sort!(A; prefer_threads, dims=2)
    @test Array(A) == sort(A_h; dims=2)

    # dims=1 on a vector is a full sort
    v_h = rand(Int32, 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sort(v; prefer_threads, dims=1)) == sort(v_h)

    # Singleton slice dimension is a no-op
    A_h = rand(Float32, 1, 64)
    A   = array_from_host(A_h)
    @test Array(AK.sort(A; prefer_threads, dims=1)) == A_h

    # Out-of-range dimension errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sort(A; prefer_threads, dims=3)
    @test_throws ArgumentError AK.sort(A; prefer_threads, dims=0)
end


@testset "sortperm_dims" begin
    Random.seed!(0)

    # Fuzzy correctness against Base.sortperm(A; dims); small integer ranges give many ties, so
    # matching Base's index array exactly also checks that the permutation is stable
    for _ in 1:200
        nd = rand(2:3)
        sz = ntuple(_ -> rand(1:15), nd)
        for T in valid_backend_eltypes(BACKEND, (Int32, Float32))
            A_h = T <: Integer ? rand(T(0):T(4), sz...) : rand(T, sz...)
            A   = array_from_host(A_h)
            for dim in 1:nd, rev in (false, true)
                ix = Array(AK.sortperm(A; prefer_threads, dims=dim, rev=rev))
                @test ix == sortperm(A_h; dims=dim, rev=rev)
                @test A_h[ix] == sort(A_h; dims=dim, rev=rev)
            end
        end
    end

    # by= and order= act on the values within each slice
    A_h = rand(Float32, 17, 23)
    A   = array_from_host(A_h)
    @test Array(AK.sortperm(A; prefer_threads, dims=1, by=x->-x)) == sortperm(A_h; dims=1, by=x->-x)
    @test Array(AK.sortperm(A; prefer_threads, dims=2,
                            order=Base.Order.Reverse)) == sortperm(A_h; dims=2, order=Base.Order.Reverse)

    # In-place fills ix with the same global linear indices as Base
    A_h = rand(Int32(0):Int32(5), 40, 31)
    A   = array_from_host(A_h)
    ix  = array_from_host(zeros(Int, 40, 31))
    AK.sortperm!(ix, A; prefer_threads, dims=2)
    @test Array(ix) == sortperm(A_h; dims=2)

    # dims=1 on a vector is a full sortperm
    v_h = rand(Int32(0):Int32(9), 5000)
    v   = array_from_host(v_h)
    @test Array(AK.sortperm(v; prefer_threads, dims=1)) == sortperm(v_h)

    # Singleton slice dimension yields the identity index array
    A_h = rand(Float32, 1, 64)
    A   = array_from_host(A_h)
    @test Array(AK.sortperm(A; prefer_threads, dims=1)) == reshape(1:64, 1, 64)

    # Out-of-range dimension errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=3)
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=0)
end
end
