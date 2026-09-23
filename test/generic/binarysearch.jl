@testset "searchsorted" begin

    Random.seed!(0)

    # Fuzzy correctness testing against Base, applied to each query
    for _ in 1:100, T in (Int32, Float32)
        num_elems_v = rand(1:100_000)
        num_elems_x = rand(1:100_000)
        vh = sort(rand(T, num_elems_v))
        xh = rand(T, num_elems_x)
        v = array_from_host(vh)
        x = array_from_host(xh)

        ix = similar(x, Int32)
        @test AK.searchsortedfirst!(ix, v, x) === ix
        @test Array(ix) == [searchsortedfirst(vh, e) for e in xh]

        ix = similar(x, Int32)
        @test AK.searchsortedlast!(ix, v, x) === ix
        @test Array(ix) == [searchsortedlast(vh, e) for e in xh]
    end

    # Orderings, as in Base
    vh = rand(Int32(-1000):Int32(1000), 10_000)
    xh = rand(Int32(-1000):Int32(1000), 1000)
    for kw in ((rev=true,), (order=Base.Order.Reverse,), (by=abs,), (lt=(>),),
               (by=abs, rev=true), (lt=(>), rev=true))
        svh = sort(vh; kw...)
        v = array_from_host(svh)
        x = array_from_host(xh)
        ix = array_from_host(zeros(Int, length(xh)))
        AK.searchsortedfirst!(ix, v, x; kw...)
        @test Array(ix) == [searchsortedfirst(svh, e; kw...) for e in xh]
        AK.searchsortedlast!(ix, v, x; kw...)
        @test Array(ix) == [searchsortedlast(svh, e; kw...) for e in xh]
    end

    # Launch settings
    v = array_from_host(sort(rand(Int32, 100_000)))
    x = array_from_host(rand(Int32, 10_000))
    ix = similar(x, Int32)
    AK.searchsortedfirst!(ix, v, x; block_size=64, max_tasks=10, min_elems=100)
    @test Array(ix) == [searchsortedfirst(Array(v), e) for e in Array(x)]
    AK.searchsortedlast!(ix, v, x; block_size=64, max_tasks=10, min_elems=100)
    @test Array(ix) == [searchsortedlast(Array(v), e) for e in Array(x)]

    # Invalid arguments
    @test_throws ArgumentError AK.searchsortedfirst!(similar(x, Int32, 3), v, x)
    @test_throws ArgumentError AK.searchsortedfirst!(ix, v, x; block_size=0)
    @test_throws MethodError AK.searchsortedfirst!(ix, v, x; bad=:kwarg)
    @test_throws MethodError AK.searchsortedlast!(ix, v, x; bad=:kwarg)
end
