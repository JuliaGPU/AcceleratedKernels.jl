if !IS_CPU_BACKEND || !prefer_threads
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
    for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND), (Float32, Float64, Int32))
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


if !IS_CPU_BACKEND || !prefer_threads
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


if !IS_CPU_BACKEND || !prefer_threads
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


if !IS_CPU_BACKEND || !prefer_threads
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


if !IS_CPU_BACKEND || !prefer_threads
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

    for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND), (Int16, UInt16, Int64, UInt64, Float64, UInt8))
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

    for (kw, check_kw) in (
        ((rev=true,),          (rev=true,)),
        ((by=abs,),            (by=abs,)),
        ((by=abs, rev=true),   (by=abs, rev=true)),
        ((lt=(>),),            (lt=(>),)),
    )
        v  = array_from_host(randn(Float32, n))
        ix = array_from_host(zeros(Int, n))
        AK.sortperm!(ix, v; kw...)
        vh, ixh = Array(v), Array(ix)
        @test is_valid_perm(vh, ixh; check_kw...)
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
    for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND), (Int32, Float32, Float64))
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

@testset "radix_sort" begin
    if !IS_CPU_BACKEND || !prefer_threads
        Random.seed!(0)

        # ── Correctness: fuzzy testing across supported types ─────────────────
        for T in (UInt32, Int32, Float32)
            for _ in 1:200
                n = rand(1:100_000)
                v = array_from_host(rand(T, n))
                AK.radix_sort!(v)
                @test issorted(Array(v))
            end
        end

        for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND),
                        (UInt64, Int64, Float64))
            for _ in 1:200
                n = rand(1:100_000)
                v = array_from_host(rand(T, n))
                AK.radix_sort!(v)
                @test issorted(Array(v))
            end
        end

        # ── Exact match against Base.sort ─────────────────────────────────────
        for T in filter(T -> T !== Float64 || KernelAbstractions.supports_float64(BACKEND),
                        (UInt32, Int32, Float32, UInt64, Int64, Float64))
            n   = 10_000
            v_h = rand(T, n)
            v   = array_from_host(v_h)
            AK.radix_sort!(v)
            @test Array(v) == sort(v_h)
        end

        # ── rev=true ──────────────────────────────────────────────────────────
        for T in (UInt32, Int32, Float32)
            n   = 10_000
            v_h = rand(T, n)
            v   = array_from_host(v_h)
            AK.radix_sort!(v; rev=true)
            @test Array(v) == sort(v_h; rev=true)
        end

        # ── Stability: equal keys → result matches sort (radix is stable) ───
        n   = 10_000
        v_h = Int32.(mod.(1:n, 100))   # 100 distinct values, 100 copies each
        v   = array_from_host(v_h)
        AK.radix_sort!(v)
        @test Array(v) == sort(v_h)

        # ── Edge cases ────────────────────────────────────────────────────────
        @test length(Array(AK.radix_sort!(array_from_host(Int32[])))) == 0
        @test Array(AK.radix_sort!(array_from_host(Int32[42]))) == Int32[42]
        @test Array(AK.radix_sort!(array_from_host(Int32[2, 1]))) == Int32[1, 2]

        # ── temp kwarg: preallocated buffer ───────────────────────────────────
        n    = 50_000
        v_h  = rand(Float32, n)
        v    = array_from_host(v_h)
        temp = similar(v)
        AK.radix_sort!(v; temp)
        @test Array(v) == sort(v_h)

        # ── Out-of-place ──────────────────────────────────────────────────────
        n   = 10_000
        v_h = rand(Float32, n)
        v   = array_from_host(v_h)
        w   = AK.radix_sort(v)
        @test Array(w) == sort(v_h)
        @test Array(v) == v_h   # input unchanged

        # ── Non-default block size ─────────────────────────────────────────────
        n   = 10_000
        v_h = rand(UInt32, n)
        v   = array_from_host(v_h)
        AK.radix_sort!(v; block_size=128)
        @test Array(v) == sort(v_h)
    end
end
end
