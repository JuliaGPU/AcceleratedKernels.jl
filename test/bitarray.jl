# `BitArray`s live on the host but have no `get_backend`: operations on them run on the host
# backend, and those writing to one use a single task, as tasks writing neighbouring elements race
# on the 64-bit chunk that holds them

@testset "BitArray: backend" begin
    v = BitVector(rand(Bool, 100))
    for x in (v, BitMatrix(rand(Bool, 10, 10)), view(v, 3:50), reshape(v, 10, 10), transpose(v),
              view(reshape(v, 10, 10), :, 2:4))
        @test AK.resolve_backend(nothing, x) == AK.HOST_BACKEND
    end
    @test AK.resolve_backend(nothing, zeros(100), v) == AK.HOST_BACKEND
end


@testset "BitArray: operations" begin
    Random.seed!(0)
    for n in (1, 63, 64, 65, 1000)
        h = rand(Bool, n)
        v = BitVector(h)
        @test AK.sum(v) == sum(h)
        @test AK.count(v) == count(h)
        @test AK.any(identity, v) == any(h)
        @test AK.all(identity, v) == all(h)
        @test AK.minimum(v) == minimum(h)
        @test AK.findall(v) == findall(h)
        @test AK.cumsum(v) == cumsum(h)
        @test AK.accumulate(|, v) == accumulate(|, h)
        @test AK.sort(v) == sort(h)
        @test AK.sort!(copy(v)) == sort(h)
        @test AK.sortperm(v) == sortperm(h)
        @test AK.reverse(v) == reverse(h)
        @test AK.map(!, v) == .!h
        @test AK.sum(view(v, 1:cld(n, 2))) == sum(h[1:cld(n, 2)])
    end
    m = BitMatrix(rand(Bool, 37, 41))
    @test AK.sum(m; dims=1) == sum(collect(m); dims=1)
    @test AK.any(identity, reshape(view(m, :), 41, 37)) == any(m)
    @test AK.sort(m; dims=1) == sort(collect(m); dims=1)
end


# A vector whose axes start at `1 + offset`, like an `OffsetArray`
struct OffsetBits{T, A <: AbstractVector{T}} <: AbstractVector{T}
    parent::A
    offset::Int
end
Base.parent(x::OffsetBits) = x.parent
Base.axes(x::OffsetBits) = (Base.IdentityUnitRange(axes(x.parent, 1) .+ x.offset),)
Base.size(x::OffsetBits) = size(x.parent)
Base.@propagate_inbounds Base.getindex(x::OffsetBits, i::Int) = x.parent[i - x.offset]
Base.@propagate_inbounds Base.setindex!(x::OffsetBits, v, i::Int) = (x.parent[i - x.offset] = v)

@testset "BitArray: writes from one task" begin
    # Many tasks on small arrays put the boundaries between tasks inside chunks; count the
    # corrupted results of each operation
    corrupted(f) = count(_ -> !f(BitVector(rand(Bool, 1000))), 1:200)
    kw = (; max_tasks=8, min_elems=1)
    part = AK.CPUThreads.Partitioned(max_tasks=8, min_elems=2)
    sample = AK.CPUThreads.SampleSort(max_tasks=8, min_elems=1)
    @test corrupted(v -> AK.map!(!, falses(1000), v; kw...) == .!v) == 0
    @test corrupted(v -> AK.map(!, v; kw...) == .!v) == 0
    @test corrupted(v -> AK.reverse!(copy(v); kw...) == reverse(v)) == 0
    @test corrupted(v -> AK.reverse!(falses(1000), v; kw...) == reverse(v)) == 0
    @test corrupted(v -> AK.accumulate!(|, falses(1000), v; alg=part) == accumulate(|, v)) == 0
    @test corrupted(v -> AK.sort!(copy(v); alg=sample) == sort(v)) == 0
    @test corrupted(v -> AK.sort!(reshape(copy(v), 40, 25); dims=1, alg=sample) ==
                         sort(reshape(v, 40, 25); dims=1)) == 0
    @test corrupted() do v
        k = rand(1:3, 1000)
        AK.sort_by_key!(copy(k), copy(v); alg=sample)[2] == v[sortperm(k)]
    end == 0
    @test corrupted() do v
        m = reshape(v, 40, 25)
        AK.mapreducedim!(identity, |, falses(1, 25), m; init=false, alg=part) == any(m; dims=1)
    end == 0
    @test corrupted(v -> AK.searchsortedfirst!(copy(v), Int[], zeros(Int, 1000); kw...) ==
                         trues(1000)) == 0
    # (a `BitVector` result, for `Bool` items)
    @test corrupted(v -> AK.findall(v; items=fill(true, 1000), alg=part) == trues(count(v))) == 0
    # Through a wrapper with offset axes, the scans copy their results in on a loop of their own
    @test corrupted() do v
        dst = OffsetBits(falses(1000), 5)
        AK.accumulate!((_, b) -> b, dst, OffsetBits(Int.(v), 5); alg=part)
        parent(dst) == v
    end == 0
    if AK.runs_kernels(AK.HOST_BACKEND)
        # Kernel algorithms on the host write degenerate shapes on Julia threads
        @test corrupted(v -> AK.accumulate!(|, copy(v), trues(1000); dims=2,
                                            alg=AK.SliceScan()) == trues(1000)) == 0
        @test corrupted(v -> AK.mapreducedim!(identity, |, copy(v), trues(1000); overwrite=true,
                                              alg=AK.BlockReduce()) == trues(1000)) == 0
    end
end
