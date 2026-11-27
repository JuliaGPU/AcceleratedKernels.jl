const SELECT_ALG = default_alg(AK.RadixSelect())

# An element type without radix keys, ordered by `isless`
struct Wrapped
    x::Int32
end
Base.isless(a::Wrapped, b::Wrapped) = isless(a.x, b.x)
Base.zero(::Type{Wrapped}) = Wrapped(0)

# The elements at `ranks` of the sorted array (dims=:) or of each sorted slice
function ref_select(h, ranks, dims, rev)
    dims isa Colon && return sort(vec(h); rev)[collect(ranks)]
    return selectdim(sort(h; dims, rev), dims, collect(ranks))
end

function select_dst(src, nr, dims)
    sz = dims isa Colon ? (nr,) : ntuple(d -> d == dims ? nr : size(src, d), ndims(src))
    return array_from_host(Array{eltype(src)}(undef, sz))
end

# Select from a copy of `h` on the backend, checking the result and that the source is untouched
# (`ranks` is a tuple here for brevity, which `select!` takes as a vector)
function check_select(h, ranks; dims=Colon(), rev=false, alg=SELECT_ALG)
    src = array_from_host(h)
    dst = select_dst(h, length(ranks), dims)
    AK.select!(dst, src, collect(Int, ranks); dims, rev, alg)
    ok = isequal(Array(dst), ref_select(h, ranks, dims, rev)) && isequal(Array(src), h)
    ok || @error "select! mismatch" eltype(h) size(h) ranks dims rev alg
    return ok
end

@testset "select" begin
    Random.seed!(0)
    radix_types = valid_backend_eltypes(BACKEND, (Int32, UInt32, Float32, Int64, UInt64, Float64))

    @testset "vectors: $T" for T in radix_types
        # lengths for each kernel: counting (<= 160), one block (<= 8192), global passes
        for n in (1, 2, 7, 100, 128, 160, 161, 1000, 8192, 8193, 50_000, 300_000)
            h = rand(T, n)
            m = cld(n, 2)
            for ranks in ((m,), (m, min(m + 1, n)), (1,), (n,), (min(m + 1, n), m), (1, n)),
                rev in (false, true)
                @test check_select(h, ranks; rev)
            end
        end
    end

    @testset "data: $T" for T in radix_types, n in (100, 5000, 100_000)
        m = n ÷ 2
        datasets = Any[
            fill(T(3), n),                                  # all equal
            T.(rand(1:3, n)),                               # few distinct
            shuffle!(T.(vcat(fill(1, m), fill(2, n - m)))), # the median between two values
        ]
        # long common prefixes: the keys agree on all leading digits but the last ones
        if T <: AbstractFloat
            push!(datasets, T(1) .+ rand(T, n) .* T(2)^-12)
            push!(datasets, T(-1000) .- rand(T, n) .* eps(T(1000)) .* 16)
        else
            push!(datasets, T(typemax(T) ÷ 3) .+ T.(rand(0:255, n)))
        end
        # the median ranks on both sides of a bucket boundary of every digit
        # (built from radix keys in the middle of the key range: finite floats)
        K = AK.select_key_type(T)
        for shift in 8:8:8 * sizeof(T) - 8
            k = K(1) << (8 * sizeof(T) - 1) + K(1) << shift - K(1)
            lo, hi = AK.select_value(T, k, false), AK.select_value(T, k + K(1), false)
            push!(datasets, shuffle!(vcat(fill(lo, m), fill(hi, n - m))))
        end
        for h in datasets, ranks in ((m,), (m, m + 1), (m + 1, m), (1,), (n,)), rev in (false, true)
            @test check_select(h, ranks; rev)
        end
    end

    @testset "floats: $T" for T in filter(T -> T <: AbstractFloat, radix_types), n in (50, 3000, 40_000)
        # signed zeros, infinities and NaNs order like `isless`
        h = rand(T[-0.0, 0.0, -Inf, Inf, NaN, 1, -1], n)
        for ranks in ((1,), (n,), (n ÷ 2, n ÷ 2 + 1), (count(!isnan, h),), (min(count(!isnan, h) + 1, n),)),
            rev in (false, true)
            @test check_select(h, ranks; rev)
        end
        h = fill(T(NaN), n)
        @test check_select(h, (1, n))
        h = rand(T[-0.0, 0.0], n)
        @test check_select(h, (count(signbit, h),))
        @test check_select(h, (count(signbit, h) + 1,))
    end

    @testset "dims: $T" for T in valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))
        for sz in ((300, 40), (40, 300), (20_000, 3), (3, 20_000), (1, 500), (500, 1),
                   (5, 200, 3), (9, 4, 2000), (64, 64, 2)), dims in 1:length(sz)
            h = rand(T, sz...)
            len = sz[dims]
            m = cld(len, 2)
            for ranks in ((m,), (m, min(m + 1, len)), (1,), (len,)), rev in (false, true)
                @test check_select(h, ranks; dims, rev)
            end
        end
        # duplicates within the slices
        h = T.(rand(1:4, 1000, 7))
        @test check_select(h, (500, 501); dims=1)
        @test check_select(h, (4,); dims=2)
        # dims=: on arrays of several dimensions selects from all elements
        h = rand(T, 30, 40, 5)
        @test check_select(h, (3000, 3001))
    end

    @testset "fallback" begin
        sort_alg = default_alg(AK.MergeSort())
        alg = AK.SortSelect(; sort_alg)
        # element types without radix keys, more than two ranks
        h = rand(Float16, 1000)
        @test check_select(h, (500, 501); alg)
        @test check_select(h, (1, 1000, 3, 3, 999); alg, rev=true)
        h = Wrapped.(rand(Int32(-50):Int32(50), 300))
        @test check_select(h, (150, 151, 1); alg)
        h = rand(Float32, 200, 30)
        @test check_select(h, (1, 100, 200, 100); dims=1, alg)
        @test check_select(h, (7, 2, 30); dims=2, alg, rev=true)
        h = rand(Float32, 5000)
        @test check_select(h, collect(1:7:5000); alg)
        if !HOST_KERNELS
            # `Auto` picks the fallback where it must
            @test check_select(rand(Float16, 777), (388, 389); alg=AK.Auto())
            @test check_select(rand(Int32, 777), (1, 2, 3); alg=AK.Auto())
            @test check_select(rand(Float32, 50, 60), [5, 9, 25]; dims=1, alg=AK.Auto())
        end
    end

    @testset "select" begin
        h = rand(Float32, 3000, 5)
        src = array_from_host(h)
        r = AK.select(src, [1500, 1501]; alg=SELECT_ALG)
        @test r isa AbstractVector{Float32} && Array(r) == ref_select(h, (1500, 1501), Colon(), false)
        r = AK.select(src, [2]; dims=2, rev=true, alg=SELECT_ALG)
        @test size(r) == (3000, 1) && Array(r) == ref_select(h, (2,), 2, true)
        @test AK.plan(AK.select, src, [7]; dims=1) isa AK.Plan
    end

    @testset "arguments" begin
        h = rand(Float32, 5000)
        src = array_from_host(h)
        # ranks with other axes and integer types, in their order
        r = AK.select(src, OffsetAxes(Int16[8, 1, 3], (-7,)))
        @test Array(r) == ref_select(h, (8, 1, 3), Colon(), false)
        r = AK.select(src, view([4000, 9, 7], 1:2:3); alg=SELECT_ALG)
        @test Array(r) == ref_select(h, (4000, 7), Colon(), false)
        # a strided destination
        D = array_from_host(zeros(Float32, 4, 3))
        AK.select!(view(D, 2, :), src, [1, 2500, 5000])
        AK.select!(view(D, 3, 1:2), src, [3, 2]; alg=SELECT_ALG)
        @test Array(D)[2, :] == ref_select(h, (1, 2500, 5000), Colon(), false)
        @test Array(D)[3, :] == [ref_select(h, (3, 2), Colon(), false); 0]
        @test Array(D)[[1, 4], :] == zeros(Float32, 2, 3)
        # one workspace for several ranks and data, whose global passes skip and compact
        # differently: their state is reset per call
        n = 50_000
        big = array_from_host(rand(Float32, n))
        dst = select_dst(big, 2, Colon())
        ws = AK.workspace(AK.select!, dst, big, [1, 2]; alg=SELECT_ALG)
        ws.plan.alg isa AK.RadixSelect && @test length(ws.buffers.comp) > 0
        for (data, ranks) in ((rand(Float32, n), [n ÷ 2, n ÷ 2 + 1]), (fill(1f0, n), [1, n]),
                              (Float32.(rand(1:3, n)), [n - 1, 2]),
                              (1f0 .+ rand(Float32, n) .* 2f0^-12, [7, n ÷ 3]),
                              (rand(Float32, n), [1, n]))
            copyto!(big, data)
            AK.select!(dst, big, ranks; alg=SELECT_ALG, workspace=ws)
            @test Array(dst) == ref_select(data, Tuple(ranks), Colon(), false)
        end
        if !HOST_KERNELS
            # what `Auto` picks: radix select on GPUs, while it can
            p = AK.plan(AK.select!, dst, src, [1, 2])
            @test p.alg isa (TEST_KERNELS ? AK.RadixSelect : AK.SortSelect)
            @test AK.plan(AK.select!, select_dst(h, 3, Colon()), src, [1, 2, 3]).alg isa AK.SortSelect
        end
    end

    @testset "inference" begin
        # the result infers, also where `Auto()` picks the algorithm from a GPU's tuning
        src = array_from_host(rand(Float32, 1000, 4))
        dst = select_dst(src, 2, 1)
        @test @inferred(AK.select!(dst, src, [10, 11]; dims=1)) === dst
        @test @inferred(AK.select(src, Int32[3]; dims=2)) isa typeof(src)
        for (alg, T) in ((AK.Auto, Float32), (AK.Auto, Wrapped), (AK.RadixSelect, Float32),
                         (AK.SortSelect{AK.Auto}, Int64))
            rt = only(Base.return_types(AK.resolve_select, (alg, typeof(BACKEND), Type{T}, Int, Int)))
            @test rt isa Union || isconcretetype(rt)
        end
    end

    @testset "workspace" begin
        h = rand(Float32, 100_000)
        src = array_from_host(h)
        dst = select_dst(h, 2, Colon())
        with_workspace(AK.select!, dst, src, [50_000, 50_001]; alg=SELECT_ALG)
        @test Array(dst) == ref_select(h, (50_000, 50_001), Colon(), false)
        h = rand(Float32, 20_000, 3)
        src = array_from_host(h)
        dst = select_dst(h, 1, 1)
        with_workspace(AK.select!, dst, src, [10_000]; dims=1, alg=SELECT_ALG)
        @test Array(dst) == ref_select(h, (10_000,), 1, false)
        r = with_workspace(AK.select, src, [3, 2]; dims=2, alg=SELECT_ALG)
        @test Array(r) == ref_select(h, (3, 2), 2, false)
    end

    @testset "errors" begin
        src = array_from_host(rand(Float32, 100))
        dst = select_dst(src, 2, Colon())
        @test_throws ArgumentError AK.select!(dst, src, [0, 1]; alg=SELECT_ALG)
        @test_throws ArgumentError AK.select!(dst, src, [1, 101]; alg=SELECT_ALG)
        # ranks are a vector of integers
        @test_throws MethodError AK.select!(dst, src, [1.0, 2.0]; alg=SELECT_ALG)
        @test_throws MethodError AK.select!(dst, src, (1, 2); alg=SELECT_ALG)
        @test_throws MethodError AK.select!(dst, src, 1; alg=SELECT_ALG)
        @test_throws ArgumentError AK.select!(select_dst(rand(Int32, 2), 2, Colon()), src,
                                              [1, 2]; alg=SELECT_ALG)
        @test_throws ArgumentError AK.select!(view(src, 1:2), src, [1, 2]; alg=SELECT_ALG)
        @test_throws DimensionMismatch AK.select!(dst, src, [1]; alg=SELECT_ALG)
        A = array_from_host(rand(Float32, 10, 20))
        @test_throws DimensionMismatch AK.select!(select_dst(A, 2, 2), A, [1, 2]; dims=1)
        @test_throws DimensionMismatch AK.select!(select_dst(A, 2, Colon()), A, [1, 2]; dims=1)
        @test_throws ArgumentError AK.select!(select_dst(A, 1, 1), A, [11]; dims=1)
        @test_throws ArgumentError AK.select!(select_dst(A, 1, 1), A, [1]; dims=3)
        @test_throws ArgumentError AK.select(A, [1]; dims=0)
        @test_throws ArgumentError AK.select!(dst, src, [1, 2]; alg=AK.MergeSort())
        if TEST_KERNELS
            h = rand(Float16, 10)
            @test_throws ArgumentError AK.select!(select_dst(h, 1, Colon()), array_from_host(h),
                                                  [1]; alg=AK.RadixSelect())
            @test_throws ArgumentError AK.select!(select_dst(src, 3, Colon()), src, [1, 2, 3];
                                                  alg=AK.RadixSelect())
            @test_throws ArgumentError AK.select!(dst, src, [1, 2];
                                                  alg=AK.RadixSelect(block_size=128))
            @test_throws ArgumentError AK.select!(dst, src, [1, 2];
                                                  alg=AK.RadixSelect(items_per_thread=64))
        end
        # empty selections
        @test AK.select!(select_dst(src, 0, Colon()), src, Int[]; alg=SELECT_ALG) isa AbstractVector
        E = array_from_host(rand(Float32, 0, 4))
        @test_throws ArgumentError AK.select!(select_dst(E, 1, 1), E, [1]; dims=1)
        @test size(AK.select!(select_dst(E, 1, 2), E, [1]; dims=2)) == (0, 1)
        # ... which need no scratch
        no_scratch(sz) = Base.all(b -> b isa NamedTuple ? no_scratch(b) : Base.prod(b[2]) == 0,
                                  values(sz))
        big = array_from_host(rand(Float32, 100_000))
        @test no_scratch(AK.workspace_size(AK.select!, select_dst(big, 0, Colon()), big, Int[]))
        @test no_scratch(AK.workspace_size(AK.select!, select_dst(big, 0, Colon()), big, Int[];
                                           alg=AK.SortSelect()))
        @test no_scratch(AK.workspace_size(AK.select!, select_dst(E, 1, 2), E, [1]; dims=2))
    end
end
