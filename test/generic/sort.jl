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

end


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


@testset "sortperm_dims" begin
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
                ix = Array(AK.sortperm(A; prefer_threads, dims=dim, rev))
                @test ix == sortperm(A_h; dims=dim, rev)
                @test A_h[ix] == sort(A_h; dims=dim, rev)
            end
        end
    end

    # Ties across block tiles and global merge passes must stay stable
    for len in (511, 512, 513, 1025, 2049, 10_000), nslices in (1, 3)
        A_h = rand(Int32(0):Int32(3), len, nslices)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=1)) == sortperm(A_h; dims=1)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=1, rev=true)) == sortperm(A_h; dims=1, rev=true)
        A_h = rand(Int32(0):Int32(3), nslices, len)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=2)) == sortperm(A_h; dims=2)
    end

    # by, order, temp and the low-memory GPU path
    A_h = rand(Float32, 300, 700)
    A   = array_from_host(A_h)
    @test Array(AK.sortperm(A; prefer_threads, dims=1, by=x->-x)) == sortperm(A_h; dims=1, by=x->-x)
    @test Array(AK.sortperm(A; prefer_threads, dims=2, order=Base.Order.Reverse)) == sortperm(A_h; dims=2, order=Base.Order.Reverse)
    @test Array(AK.sortperm(A; prefer_threads, dims=2, temp=similar(A, Int))) == sortperm(A_h; dims=2)
    if !prefer_threads
        @test Array(AK.sortperm(A; prefer_threads, dims=2, alg=AK.MergeSort(lowmem=true))) == sortperm(A_h; dims=2)
        @test Array(AK.sortperm(A; prefer_threads, dims=1, alg=AK.MergeSort(lowmem=true), block_size=64)) == sortperm(A_h; dims=1)
    end

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

    # Empty and singleton slices
    for sz in ((0, 5), (5, 0), (1, 64), (64, 1)), dim in 1:2
        A_h = rand(Float32, sz...)
        @test Array(AK.sortperm(array_from_host(A_h); prefer_threads, dims=dim)) == sortperm(A_h; dims=dim)
    end

    # Slice offsets refer to the view's linear indexing, not its parent's strides.
    A_h = rand(Int32(0):Int32(3), 6, 1030)
    A = array_from_host(A_h)
    V_h = view(A_h, 1:2:6, 1:2:1030)
    V = view(A, 1:2:6, 1:2:1030)
    @test Array(AK.sortperm(V; prefer_threads, dims=2)) == sortperm(V_h; dims=2)
    AK.sort!(V; prefer_threads, dims=2)
    sort!(V_h; dims=2)
    @test Array(A) == A_h

    # Floating-point ordering also applies to the low-level entry points.
    A_h = Float32[NaN 1 -0.0; 0.0 -Inf NaN; 2 NaN Inf]
    A = array_from_host(A_h)
    for dim in 1:2, rev in (false, true)
        expected = sortperm(A_h; dims=dim, rev)
        @test Array(AK.sortperm(A; prefer_threads, dims=dim, rev)) == expected
        if !prefer_threads
            @test Array(AK.merge_sortperm(A; dims=dim, rev)) == expected
            @test Array(AK.merge_sortperm_lowmem(A; dims=dim, rev)) == expected
        end
    end

    # Invalid dimensions must not overwrite the output.
    ix = array_from_host(fill(-1, size(A_h)))
    @test_throws ArgumentError AK.sortperm!(ix, A; prefer_threads, dims=3)
    @test Array(ix) == fill(-1, size(A_h))
    if !prefer_threads
        @test_throws ArgumentError AK.merge_sortperm!(ix, A; dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.merge_sortperm_lowmem!(ix, A; dims=3)
        @test Array(ix) == fill(-1, size(A_h))
        @test_throws ArgumentError AK.merge_sort_by_key!(copy(A), similar(ix, length(ix)); dims=1)
    end

    # Out-of-range dimension and mismatched index array errors
    A = array_from_host(rand(Float32, 8, 8))
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=3)
    @test_throws ArgumentError AK.sortperm(A; prefer_threads, dims=0)
    @test_throws ArgumentError AK.sortperm!(array_from_host(zeros(Int, 64)), A; prefer_threads, dims=1)
end


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
