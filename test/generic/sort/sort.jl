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
