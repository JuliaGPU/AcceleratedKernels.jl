struct FindallCallable end
(::FindallCallable)(x) = x > zero(x)

struct OtherFindallAlgorithm <: AK.FindallAlgorithm end

struct FindallOffsetVector{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
    offset::Int
end

Base.size(v::FindallOffsetVector) = size(v.data)
Base.axes(v::FindallOffsetVector) =
    (Base.IdentityUnitRange((firstindex(v.data) + v.offset):(lastindex(v.data) + v.offset)),)
Base.IndexStyle(::Type{<:FindallOffsetVector}) = IndexLinear()
Base.getindex(v::FindallOffsetVector, i::Int) = v.data[i - v.offset]
Base.setindex!(v::FindallOffsetVector, x, i::Int) = (v.data[i - v.offset] = x)
Base.similar(v::FindallOffsetVector, ::Type{T}) where T =
    FindallOffsetVector(similar(v.data, T), v.offset)
Base.similar(v::FindallOffsetVector, ::Type{T}, dims::Dims) where T = similar(v.data, T, dims)


@testset "findall" begin
    Random.seed!(0)

    default_alg = AK.ScanScatter()
    tile_size = default_alg.block_size * default_alg.items_per_thread
    edge_sizes = [0, 1, 2, 3, tile_size - 1, tile_size, tile_size + 1,
                  2tile_size - 1, 2tile_size, 2tile_size + 1, 10_000]
    test_types = valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))

    @testset "predicate" begin
        for T in test_types, n in edge_sizes
            pred = T <: AbstractFloat ? (x -> x > oftype(x, 0.5)) : (x -> x > zero(x))
            h = T <: AbstractFloat ? rand(T, n) : rand(T(-5):T(5), n)
            v = array_from_host(h)
            @test Array(AK.findall(pred, v; prefer_threads)) == findall(pred, h)
        end

        h = collect(Int32, -10:10)
        v = array_from_host(h)
        @test Array(AK.findall(FindallCallable(), v; prefer_threads)) == findall(x -> x > 0, h)

        if prefer_threads
            calls = Ref(0)
            pred = x -> (calls[] += 1; isodd(x))
            h = collect(1:100)
            @test AK.findall(pred, h; max_tasks=1) == findall(isodd, h)
            @test calls[] == length(h)
            @test_throws TypeError AK.findall(Returns(1), [1]; max_tasks=1)
        end
    end

    @testset "mask" begin
        for n in edge_sizes
            h = rand(Bool, n)
            v = array_from_host(h)
            out = AK.findall(v; prefer_threads)
            @test Array(out) == findall(h)
            @test eltype(out) == Int
        end

        if prefer_threads
            @test AK.findall(Any[true, false, true]) == findall(Any[true, false, true])
            @test AK.findall(Any[true, false, true]; temp_bools=Vector{Bool}(undef, 3)) ==
                  findall(Any[true, false, true])
            @test_throws TypeError AK.findall([1])
            @test_throws TypeError AK.findall(Any[true, missing]; max_tasks=1)

            scalar = Array{Any}(undef)
            scalar[] = true
            @test AK.findall(scalar) == findall(scalar)
        end
    end

    @testset "dimensions and keys" begin
        for shape in ([4, 2], [1, 6], [64, 64], [8, 8, 8])
            h = rand(Float32, shape...)
            v = array_from_host(h)
            out = AK.findall(x -> x > 0.5f0, v; prefer_threads)
            @test Array(out) == findall(x -> x > 0.5f0, h)
            @test eltype(out) == CartesianIndex{length(shape)}

            hb = rand(Bool, shape...)
            @test Array(AK.findall(array_from_host(hb); prefer_threads)) == findall(hb)
        end

        for value in (false, true)
            h = fill(value)
            @test Array(AK.findall(array_from_host(h); prefer_threads)) == findall(h)
        end
        for value in (0.25f0, 0.75f0)
            h = fill(value)
            pred = x -> x > 0.5f0
            @test Array(AK.findall(pred, array_from_host(h); prefer_threads)) == findall(pred, h)
        end

        if prefer_threads
            h = FindallOffsetVector([-1, 1, -2, 2, 0], -3)
            mask = FindallOffsetVector(Bool[false, true, true, false, true], -3)
            for max_tasks in (1, 4)
                @test AK.findall(x -> x > 0, h, BACKEND; max_tasks, min_elems=1) ==
                      [-1, 1]
                @test AK.findall(mask, BACKEND; max_tasks, min_elems=1) == findall(mask)
            end

            h = collect(1:20)
            v = @view h[2:2:20]
            @test AK.findall(isodd, v) == findall(isodd, v)
            @test AK.findall(iszero, reshape(Int[], 0, 2)) ==
                  findall(iszero, reshape(Int[], 0, 2))
        end
    end

    @testset "selection extremes" begin
        for n in (0, 1, 2, 1000), h in (trues(n), falses(n))
            values = collect(h)
            @test Array(AK.findall(array_from_host(values); prefer_threads)) == findall(values)
        end

        v = array_from_host(collect(Int32, 1:1000))
        @test Array(AK.findall(x -> x > 0, v; prefer_threads)) == collect(1:1000)
        @test Array(AK.findall(x -> x < 0, v; prefer_threads)) == Int[]
    end

    @testset "random sizes" begin
        for _ in 1:100
            n = rand(1:100_000)
            h = rand(Float32, n)
            v = array_from_host(h)
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads)) ==
                  findall(x -> x > 0.5f0, h)
        end
    end

    @testset "configuration and buffers" begin
        h = rand(Float32, 10_000)
        v = array_from_host(h)
        for block_size in (32, 64, 128, 256), items_per_thread in (1, 3, 8)
            alg = AK.ScanScatter(; block_size, items_per_thread)
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads, alg)) ==
                  findall(x -> x > 0.5f0, h)
        end

        for (max_tasks, min_elems) in ((1, 1), (2, 100), (4, 1000))
            @test Array(AK.findall(x -> x > 0.5f0, v; prefer_threads, max_tasks, min_elems)) ==
                  findall(x -> x > 0.5f0, h)
        end

        alg = AK.ScanScatter(block_size=64, items_per_thread=3)
        temp = similar(v, Int, max(4, cld(length(v), alg.block_size * alg.items_per_thread)))
        temp_bools = similar(v, Bool)
        @test Array(AK.findall(x -> x > 0.5f0, v;
                              prefer_threads, max_tasks=4, alg, temp, temp_bools)) ==
              findall(x -> x > 0.5f0, h)

        @test_throws ArgumentError AK.findall(v; prefer_threads, alg=OtherFindallAlgorithm())
        @test_throws ArgumentError AK.findall(identity, temp_bools;
                                               prefer_threads, temp_bools)
        @test_throws ArgumentError AK.findall(identity, v; prefer_threads,
                                               temp_bools=reshape(similar(v, Bool), :, 1))

        if !prefer_threads
            bools = array_from_host(rand(Bool, length(v)))
            @test_throws ArgumentError AK.findall(bools; prefer_threads,
                                                  alg=AK.ScanScatter(block_size=192))
            @test_throws ArgumentError AK.findall(bools; prefer_threads,
                                                  alg=AK.ScanScatter(items_per_thread=0))
            @test_throws ArgumentError AK.findall(bools; prefer_threads,
                                                  temp=similar(v, Int32, 100))
            @test_throws ArgumentError AK.findall(bools; prefer_threads,
                                                  temp=similar(v, Int, 1))
        end
    end
end
