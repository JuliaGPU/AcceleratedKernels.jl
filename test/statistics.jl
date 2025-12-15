@testset "mean" begin
    Random.seed!(0)

    # Test all possible corner cases against Statistics.mean
    for dims in 1:3
        for isize in 1:3
            for jsize in 1:3
                for ksize in 1:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.mean(s; prefer_threads,  dims)
                    dh = Array(d)
                    @test dh ≈ mean(sh;  dims)
                    @test eltype(dh) == eltype(mean(sh; dims)) || eltype(dh) == Float32
                    d = AK.mean(s; prefer_threads,  dims)
                    dh = Array(d)
                    @test dh ≈ mean(sh;  dims)
                    @test eltype(dh) == eltype(mean(sh; dims)) || eltype(dh) == Float32
                end
            end
        end
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mean(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ mean(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mean(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ mean(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            s = AK.mean(v; prefer_threads,  dims)
            sh = Array(s)
            @test sh ≈ mean(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:4
            n1 = rand(1:100)
            n2 = rand(1:100)
            n3 = rand(1:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.mean(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ mean(vh; dims)
        end
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.mean(array_from_host(rand(Int32, 10, 10)); prefer_threads,bad=:kwarg)

    # Testing different settings
    AK.mean(
        x->x ,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=nothing,
        block_size=64,
        temp=nothing,
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.mean(
        x->x ,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=2,
        block_size=64,
        temp=array_from_host(zeros(Float32, 3, 1, 5)),
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.mean(
        x->x ,
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=3,
        block_size=64,
        temp=array_from_host(zeros(Float32, 3, 4, 1)),
        switch_below=50,
        max_tasks=16,
        min_elems=1000,
    )
end

@testset "var" begin
    Random.seed!(0)

    # Test all possible corner cases against Statistics.var
    for dims in 1:3
        for isize in 2:3
            for jsize in 2:3
                for ksize in 2:3
                    sh = rand(Int32(1):Int32(100), isize, jsize, ksize)
                    s = array_from_host(sh)
                    d = AK.var(s; prefer_threads,  dims)
                    dh = Array(d)
                    @test dh ≈ var(sh;  dims)
                    @test eltype(dh) == eltype(var(sh; dims)) || eltype(dh) == Float32
                end
            end
        end
    end

    # Fuzzy correctness testing
    for _ in 1:100
        for dims in 1:3
            n1 = rand(2:100)
            n2 = rand(2:100)
            n3 = rand(2:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.var(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ var(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(2:100)
            n2 = rand(2:100)
            n3 = rand(2:100)
            vh = rand(UInt32(1):UInt32(100), n1, n2, n3)
            v = array_from_host(vh)
            s = AK.var(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ var(vh; dims)
        end
    end

    for _ in 1:100
        for dims in 1:3
            n1 = rand(2:100)
            n2 = rand(2:100)
            n3 = rand(2:100)
            vh = rand(Float32, n1, n2, n3)
            v = array_from_host(vh)
            s = AK.var(v; prefer_threads,  dims)
            sh = Array(s)
            @test sh ≈ var(vh; dims)
        end
    end


    for _ in 1:100
        for dims in 1:3
            n1 = rand(2:100)
            n2 = rand(2:100)
            n3 = rand(2:100)
            vh = rand(Int32(1):Int32(100), n1, n2, n3)
            v = array_from_host(vh)
            init = rand(1:100)
            s = AK.var(v; prefer_threads, dims)
            sh = Array(s)
            @test sh ≈ var(vh; dims)
        end
    end

    # Test that undefined kwargs are not accepted
    @test_throws MethodError AK.var(array_from_host(rand(Int32, 10, 10)); prefer_threads,bad=:kwarg)

    # Testing different settings
    AK.var(
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=nothing,
        corrected=true,
        block_size=64,
        temp=nothing,
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.var(
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=2,
        corrected=true,
        block_size=64,
        temp=nothing,
        switch_below=50,
        max_tasks=10,
        min_elems=100,
    )
    AK.var(
        array_from_host(rand(Int32, 3, 4, 5));
        prefer_threads,
        dims=3,
        corrected=false,
        block_size=64,
        temp = nothing,
        switch_below=50,
        max_tasks=16,
        min_elems=1000,
    )
end
