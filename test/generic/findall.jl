@testset "findall" begin

    Random.seed!(0)

    # Sizes around the block boundary plus the degenerate ones
    edge_sizes = [0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 10_000]

    test_types = valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))

    # Predicate-form against Base, 1-D, several eltypes
    @testset "findall(pred, v)" begin
        for T in test_types, n in edge_sizes
            pred = T <: AbstractFloat ? (x -> x > oftype(x, 0.5)) : (x -> x > zero(x))
            h = T <: AbstractFloat ? rand(T, n) : rand(T(-5):T(5), n)
            v = array_from_host(h)
            @test Array(AK.findall(pred, v; prefer_threads)) == findall(pred, h)
        end
    end

    # Bool-mask form against Base, 1-D
    @testset "findall(bools)" begin
        for n in edge_sizes
            h = rand(Bool, n)
            v = array_from_host(h)
            out = AK.findall(v; prefer_threads)
            @test Array(out) == findall(h)
            @test eltype(out) == Int         # vector -> Int keys, matching Base
        end
    end

    # N-D input exercises the CartesianIndex key path
    @testset "N-dimensional" begin
        for shape in ([4, 2], [1, 6], [64, 64], [8, 8, 8])
            h = rand(Float32, shape...)
            v = array_from_host(h)
            out = AK.findall(x -> x > 0.5f0, v; prefer_threads)
            @test Array(out) == findall(x -> x > 0.5f0, h)
            @test eltype(out) == CartesianIndex{length(shape)}

            hb = rand(Bool, shape...)
            @test Array(AK.findall(array_from_host(hb); prefer_threads)) == findall(hb)
        end
    end

    # Degenerate masks: empty, all-true, all-false, single element
    @testset "edge masks" begin
        for n in (0, 1, 2, 1000)
            for h in (trues(n), falses(n))
                v = array_from_host(collect(h))
                @test Array(AK.findall(v; prefer_threads)) == findall(collect(h))
            end
        end
        # all elements pass / none pass, via predicate
        v = array_from_host(collect(Int32, 1:1000))
        @test Array(AK.findall(x -> x > 0, v; prefer_threads)) == collect(1:1000)
        @test Array(AK.findall(x -> x < 0, v; prefer_threads)) == Int[]
    end

    # Randomised sweep over non-block-multiple sizes
    @testset "random sizes" begin
        for _ in 1:100
            n = rand(1:100_000)
            h = rand(Float32, n)
            v = array_from_host(h)
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads)) == findall(x -> x > 0.5f0, h)
        end
    end

    # Tuning settings must not change results
    @testset "settings" begin
        h = rand(Float32, 10_000)
        for block_size in (32, 64, 128, 256)
            v = array_from_host(h)
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads, block_size)) ==
                  findall(x -> x > 0.5f0, h)
        end
        for (max_tasks, min_elems) in ((1, 1), (2, 100), (4, 1000))
            v = array_from_host(h)
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads, max_tasks, min_elems)) ==
                  findall(x -> x > 0.5f0, h)
        end
    end
end
