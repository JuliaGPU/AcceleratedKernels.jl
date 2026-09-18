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
