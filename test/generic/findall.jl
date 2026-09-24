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


# Tests that do not choose an algorithm use `FINDALL_ALG`: `Auto()`, except in the `--cpu-ka`
# configuration, whose point is to run AK's kernels on the host backend. `findall_alg` builds an
# explicitly tuned algorithm for the configuration from both kinds of settings.
FINDALL_ALG = HOST_KERNELS ? AK.ScanScatter() : AK.Auto()
findall_alg(; block_size=nothing, items_per_thread=nothing, max_tasks=nothing, min_elems=nothing) =
    TEST_KERNELS ? AK.ScanScatter(; block_size, items_per_thread) :
                   AK.CPUThreads.Partitioned(; max_tasks, min_elems)


@testset "findall" begin
    Random.seed!(0)

    default_tuning = AK.FindallTuning()
    tile_size = default_tuning.block_size * default_tuning.items_per_thread
    edge_sizes = [0, 1, 2, 3, tile_size - 1, tile_size, tile_size + 1,
                  2tile_size - 1, 2tile_size, 2tile_size + 1, 10_000]
    test_types = valid_backend_eltypes(BACKEND, (Int32, Float32, Float64))

    @testset "predicate" begin
        for T in test_types, n in edge_sizes
            pred = T <: AbstractFloat ? (x -> x > oftype(x, 0.5)) : (x -> x > zero(x))
            h = T <: AbstractFloat ? rand(T, n) : rand(T(-5):T(5), n)
            v = array_from_host(h)
            @test Array(AK.findall(pred, v; alg=FINDALL_ALG)) == findall(pred, h)
        end

        h = collect(Int32, -10:10)
        v = array_from_host(h)
        @test Array(AK.findall(FindallCallable(), v; alg=FINDALL_ALG)) == findall(x -> x > 0, h)

        if !TEST_KERNELS
            calls = Ref(0)
            pred = x -> (calls[] += 1; isodd(x))
            h = collect(1:100)
            @test AK.findall(pred, h; alg=AK.CPUThreads.Partitioned(max_tasks=1)) == findall(isodd, h)
            @test calls[] == length(h)
            @test_throws ArgumentError AK.findall(Returns(1), [1]; alg=AK.CPUThreads.Partitioned(max_tasks=1))
        end
    end

    @testset "mask" begin
        for n in edge_sizes
            h = rand(Bool, n)
            v = array_from_host(h)
            out = AK.findall(v; alg=FINDALL_ALG)
            @test Array(out) == findall(h)
            @test eltype(out) == Int
        end

        if !TEST_KERNELS
            @test AK.findall(Any[true, false, true]) == findall(Any[true, false, true])
            @test AK.findall(Any[true, false, true]; temp_bools=Vector{Bool}(undef, 3)) ==
                  findall(Any[true, false, true])
            @test_throws ArgumentError AK.findall([1])
            @test_throws TypeError AK.findall(Any[true, missing];
                                              alg=AK.CPUThreads.Partitioned(max_tasks=1))

            scalar = Array{Any}(undef)
            scalar[] = true
            @test AK.findall(scalar) == findall(scalar)
        end
    end

    @testset "dimensions and keys" begin
        for shape in ([4, 2], [1, 6], [64, 64], [8, 8, 8])
            h = rand(Float32, shape...)
            v = array_from_host(h)
            out = AK.findall(x -> x > 0.5f0, v; alg=FINDALL_ALG)
            @test Array(out) == findall(x -> x > 0.5f0, h)
            @test eltype(out) == CartesianIndex{length(shape)}

            hb = rand(Bool, shape...)
            @test Array(AK.findall(array_from_host(hb); alg=FINDALL_ALG)) == findall(hb)
        end

        for value in (false, true)
            h = fill(value)
            @test Array(AK.findall(array_from_host(h); alg=FINDALL_ALG)) == findall(h)
        end
        # The default items are `keys(A)`, also for a 0-dimensional array and the predicate form
        for value in (0.25f0, 0.75f0)
            h = fill(value)
            pred = x -> x > 0.5f0
            out = AK.findall(pred, array_from_host(h); alg=FINDALL_ALG)
            @test eltype(out) === CartesianIndex{0} && Array(out) == findall(fill(pred(h[])))
            out = AK.findall(pred, array_from_host(h); items=LinearIndices(h), alg=FINDALL_ALG)
            @test eltype(out) === Int && Array(out) == (value > 0.5f0 ? [1] : Int[])
        end

        if !TEST_KERNELS
            # Positions are ordinal, so offset axes pair with any items
            h = FindallOffsetVector([-1, 1, -2, 2, 0], -3)
            mask = FindallOffsetVector(Bool[false, true, true, false, true], -3)
            for max_tasks in (1, 4)
                alg = AK.CPUThreads.Partitioned(; max_tasks, min_elems=1)
                @test AK.findall(x -> x > 0, h; backend=BACKEND, alg) == [-1, 1]
                @test AK.findall(mask; backend=BACKEND, alg) == findall(mask)
                @test AK.findall(x -> x > 0, h; items=10:14, backend=BACKEND, alg) == [11, 13]
                @test AK.findall(mask; items=h, backend=BACKEND, alg) == [1, -2, 0]
            end

            h = collect(1:20)
            v = @view h[2:2:20]
            @test AK.findall(isodd, v) == findall(isodd, v)
            @test AK.findall(iszero, reshape(Int[], 0, 2)) ==
                  findall(iszero, reshape(Int[], 0, 2))
        end
    end

    @testset "items" begin
        h = rand(Float32, 37, 29)
        v = array_from_host(h)
        pred = x -> x > 0.5f0
        sel = vec(pred.(h))
        # Linear indices of a matrix, and values
        @test Array(AK.findall(pred, v; items=LinearIndices(v), alg=FINDALL_ALG)) ==
              findall(sel)
        out = AK.findall(pred, v; items=v, alg=FINDALL_ALG)
        @test eltype(out) === Float32 && Array(out) == h[sel]
        hb = rand(Bool, size(h))
        @test Array(AK.findall(array_from_host(hb); items=v, alg=FINDALL_ALG)) == h[hb]
        @test Array(AK.findall(array_from_host(hb); items=CartesianIndices(v), alg=FINDALL_ALG)) ==
              findall(hb)
        # ... of another shape or on another array, paired by position
        w = array_from_host(collect(Int32, 1:length(h)))
        @test Array(AK.findall(pred, v; items=w, alg=FINDALL_ALG)) == findall(sel)
        @test Array(AK.findall(isodd, 1:10; items=array_from_host(collect(Int32, 11:20)),
                               alg=FINDALL_ALG)) == 11:2:19
        # A wrapper whose `eachindex` is Cartesian, with its own linear indices
        hv = view(h, 1:2:37, :)
        vv = view(v, 1:2:37, :)
        @test Array(AK.findall(pred, vv; items=LinearIndices(vv), alg=FINDALL_ALG)) ==
              findall(vec(pred.(hv)))
        @test Array(AK.findall(pred, vv; alg=FINDALL_ALG)) == findall(pred, hv)
        # A mask whose wrappers `@Const` cannot rebuild on the device (a reshaped view)
        hbm = rand(Bool, 8, 8)
        vbm = vec(view(array_from_host(hbm), 1:4, 1:4))
        @test Array(AK.findall(vbm; alg=FINDALL_ALG)) == findall(vec(view(hbm, 1:4, 1:4)))
        # `items` must have the array's length
        @test_throws DimensionMismatch AK.findall(pred, v; items=1:3, alg=FINDALL_ALG)
        @test_throws DimensionMismatch AK.findall(array_from_host(hb); items=1:3, alg=FINDALL_ALG)
    end

    @testset "selection extremes" begin
        for n in (0, 1, 2, 1000), h in (trues(n), falses(n))
            values = collect(h)
            @test Array(AK.findall(array_from_host(values); alg=FINDALL_ALG)) == findall(values)
        end

        # A range, with its backend given explicitly
        @test Array(AK.findall(isodd, 1:10; backend=BACKEND, alg=FINDALL_ALG)) == findall(isodd, 1:10)
        # (indexing a `Bool` range may throw an `InexactError`, which on oneAPI needs oneAPI.jl
        # 2.9.2's device heap)
        @test Array(AK.findall(false:true; backend=BACKEND, alg=FINDALL_ALG)) == findall(false:true)
        @test Array(AK.findall(true:false; backend=BACKEND, alg=FINDALL_ALG)) == Int[]

        v = array_from_host(collect(Int32, 1:1000))
        @test Array(AK.findall(x -> x > 0, v; alg=FINDALL_ALG)) == collect(1:1000)
        @test Array(AK.findall(x -> x < 0, v; alg=FINDALL_ALG)) == Int[]
    end

    @testset "random sizes" begin
        for _ in 1:100
            n = rand(1:100_000)
            h = rand(Float32, n)
            v = array_from_host(h)
            @test Array(AK.findall(x -> x > 0.5f0, v; alg=FINDALL_ALG)) ==
                  findall(x -> x > 0.5f0, h)
        end
    end

    @testset "configuration and buffers" begin
        h = rand(Float32, 10_000)
        v = array_from_host(h)
        for block_size in (32, 64, 128, 256), items_per_thread in (1, 3, 8)
            alg = findall_alg(; block_size, items_per_thread)
            @test Array(AK.findall(x -> x > 0.5f0, v; alg)) ==
                  findall(x -> x > 0.5f0, h)
        end

        for (max_tasks, min_elems) in ((1, 1), (2, 100), (4, 1000))
            @test Array(AK.findall(x -> x > 0.5f0, v; alg=findall_alg(; max_tasks, min_elems))) ==
                  findall(x -> x > 0.5f0, h)
        end

        alg = findall_alg(block_size=64, items_per_thread=3, max_tasks=4)
        temp = similar(v, Int, max(4, cld(length(v), 64 * 3)))
        temp_bools = similar(v, Bool)
        @test Array(AK.findall(x -> x > 0.5f0, v; alg, temp, temp_bools)) ==
              findall(x -> x > 0.5f0, h)

        @test_throws ArgumentError AK.findall(v; alg=OtherFindallAlgorithm())
        @test_throws ArgumentError AK.findall(identity, temp_bools;
                                               temp_bools)
        @test_throws ArgumentError AK.findall(identity, v; temp_bools=reshape(similar(v, Bool), :, 1))

        if TEST_KERNELS
            bools = array_from_host(rand(Bool, length(v)))
            @test_throws ArgumentError AK.findall(bools; alg=AK.ScanScatter(block_size=192))
            @test_throws ArgumentError AK.findall(bools; alg=AK.ScanScatter(items_per_thread=0))
            @test_throws ArgumentError AK.findall(bools; alg=AK.ScanScatter(),
                                                  temp=similar(v, Int32, 100))
            @test_throws ArgumentError AK.findall(bools; alg=AK.ScanScatter(), temp=similar(v, Int, 1))
        end
    end
end


# A GPU backend with its own tuning
struct FindallResolveTestBackend <: KernelAbstractions.GPU end
AK.findall_tuning(::FindallResolveTestBackend, ::Type) =
    AK.FindallTuning(block_size=128, items_per_thread=4)

@testset "findall resolution" begin
    B = FindallResolveTestBackend()
    @test AK._resolve_findall(AK.Auto(), B, Bool) === AK.ScanScatter(128, 4)
    @test AK._resolve_findall(AK.ScanScatter(block_size=64), B, Bool) === AK.ScanScatter(64, 4)
    @test AK._resolve_findall(AK.Auto(), AK.HOST_BACKEND, Bool) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    for bad in (AK.ScanScatter(block_size=0), AK.ScanScatter(block_size=96),
                AK.ScanScatter(block_size=2048), AK.ScanScatter(items_per_thread=0),
                AK.ScanScatter(block_size=1024, items_per_thread=1 << 22))
        @test_throws ArgumentError AK._resolve_findall(bad, B, Bool)
    end
    @test_throws ArgumentError AK._resolve_findall(AK.CPUThreads.Partitioned(), B, Bool)
    @test_throws ArgumentError AK._resolve_findall(AK.BlockReduce(), B, Bool)

    # Removed keywords
    v = array_from_host(rand(Bool, 10))
    @test_throws MethodError AK.findall(v; prefer_threads=true)
    @test_throws MethodError AK.findall(v; max_tasks=2)
    @test_throws MethodError AK.any(identity, v; prefer_threads=true)
    @test_throws MethodError AK.all(identity, v; block_size=64)
end


@testset "findall: Bool values" begin
    # The predicate (or the values) must give a `Bool`, as in Base; where inference shows they
    # cannot, before launching (not every backend reports an error thrown in a kernel)
    @test_throws ArgumentError AK.findall(x -> 1, array_from_host(Int32[1, 2]))
    @test_throws ArgumentError AK.findall(array_from_host(Int32[1, 0]))
    # ... except on an empty array, where Base never calls it
    @test isempty(AK.findall(x -> 1, array_from_host(Int32[])))
end
