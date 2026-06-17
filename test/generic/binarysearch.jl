@testset "searchsorted" begin

    Random.seed!(0)

    # Fuzzy correctness testing of searchsortedfirst
    for _ in 1:100
        num_elems_v = rand(1:100_000)
        num_elems_x = rand(1:100_000)

        # Ints
        v = array_from_host(sort(rand(Int32, num_elems_v)))
        x = array_from_host(rand(Int32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedfirst!(ix, v, x; prefer_threads)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedfirst(vh, xh; prefer_threads)
        ixh_base = [searchsortedfirst(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)

        # Floats
        v = array_from_host(sort(rand(Float32, num_elems_v)))
        x = array_from_host(rand(Float32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedfirst!(ix, v, x; prefer_threads)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedfirst(vh, xh; prefer_threads)
        ixh_base = [searchsortedfirst(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)
    end

    # Fuzzy correctness testing of searchsortedlast
    for _ in 1:100
        num_elems_v = rand(1:100_000)
        num_elems_x = rand(1:100_000)

        # Ints
        v = array_from_host(sort(rand(Int32, num_elems_v)))
        x = array_from_host(rand(Int32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedlast!(ix, v, x; prefer_threads)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedlast(vh, xh; prefer_threads)
        ixh_base = [searchsortedlast(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)

        # Floats
        v = array_from_host(sort(rand(Float32, num_elems_v)))
        x = array_from_host(rand(Float32, num_elems_x))
        ix = similar(x, Int32)
        AK.searchsortedlast!(ix, v, x; prefer_threads)

        vh = Array(v)
        xh = Array(x)
        ixh = AK.searchsortedlast(vh, xh; prefer_threads)
        ixh_base = [searchsortedlast(vh, e) for e in xh]

        @test all(Array(ix) .== ixh .== ixh_base)
    end

    # Testing different settings
    v = array_from_host(sort(rand(Int32, 100_000)))
    x = array_from_host(rand(Int32, 10_000))
    ix = similar(x, Int32)

    AK.searchsortedfirst!(ix, v, x; prefer_threads, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedfirst(v, x; prefer_threads, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedlast!(ix, v, x; prefer_threads, by=abs, lt=(>), rev=true, block_size=64)
    AK.searchsortedlast(v, x; prefer_threads, by=abs, lt=(>), rev=true, block_size=64)

    vh = Array(v)
    xh = Array(x)
    ixh = similar(xh, Int32)

    AK.searchsortedfirst!(ixh, vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedfirst(vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedlast!(ixh, vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)
    AK.searchsortedlast(vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100)

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.searchsortedfirst!(ixh, vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100, bad=:kwarg)
    @test_throws MethodError AK.searchsortedfirst(vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100, bad=:kwarg)
    @test_throws MethodError AK.searchsortedlast!(ixh, vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100, bad=:kwarg)
    @test_throws MethodError AK.searchsortedlast(vh, xh; prefer_threads, by=abs, lt=(>), rev=true, max_tasks=10, min_elems=100, bad=:kwarg)
end
