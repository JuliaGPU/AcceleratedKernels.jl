# The workspace protocol: `workspace_size` and `workspace` plan the same buffers as the operation,
# which then allocates no scratch of its own; a workspace made for another call is rejected.

@testset "workspace: sizes" begin
    v = array_from_host(rand(Float32, 10_000))
    ks = array_from_host(rand(Int32(1):Int32(9), 10_000))

    # Every operation that can need scratch has a plan; its sizes are `(eltype, dims)` pairs, or
    # nested for the operations it calls
    function check_sizes(sizes)
        @test sizes isa NamedTuple
        for s in values(sizes)
            s isa NamedTuple ? check_sizes(s) :
                @test s isa Tuple{Type, Tuple} && Base.all(d -> d isa Int && d >= 0, s[2])
        end
    end
    for (op, args, kw) in (
            (AK.sort!, (copy(v),), (;)), (AK.sort, (v,), (;)), (AK.sort!, (copy(v),), (; by=abs)),
            (AK.sortperm!, (similar(v, Int), v), (;)), (AK.sortperm, (v,), (;)),
            (AK.sort_by_key!, (copy(ks), copy(v)), (;)),
            (AK.mapreduce, (abs, +, v), (;)), (AK.reduce, (+, v), (;)), (AK.sum, (v,), (;)),
            (AK.prod, (v,), (;)), (AK.maximum, (v,), (;)), (AK.minimum, (v,), (;)),
            (AK.count, (x -> x > 0.5f0, v), (;)), (AK.count, (array_from_host(rand(Bool, 100)),), (;)),
            (AK.reduce, (+, reshape(v, 100, 100)), (; dims=1)),
            (AK.mapreducedim!, (identity, +, array_from_host(zeros(Float32, 1, 100)), reshape(v, 100, 100)), (;)),
            (AK.accumulate!, (+, copy(v)), (;)), (AK.accumulate!, (+, similar(v), v), (;)),
            (AK.accumulate, (+, v), (;)), (AK.cumsum, (v,), (;)), (AK.cumprod, (v,), (;)),
            (AK.accumulate, (+, reshape(v, 100, 100)), (; dims=2)),
            (AK.findall, (x -> x > 0.5f0, v), (;)), (AK.findall, (array_from_host(rand(Bool, 100)),), (;)),
            (AK.findall, (x -> x > 0.5f0, v), (; items=v)),
            (AK.any, (x -> x > 2, v), (;)), (AK.all, (x -> x < 2, v), (;)),
            # (the keywords that change the buffers' types)
            (AK.sum, (v,), (; acctype=Float32)), (AK.reduce, (+, reshape(v, 100, 100)), (; dims=1, acctype=Float32)),
            (AK.mapreducedim!, (identity, +, array_from_host(zeros(Float32, 1, 100)), reshape(v, 100, 100)), (; acctype=Float32)),
            (AK.accumulate!, (+, similar(ks), ks), (; acctype=Int)), (AK.accumulate, (+, v), (; acctype=Float32)),
        )
        sizes = AK.workspace_size(op, args...; kw...)
        check_sizes(sizes)
        ws = AK.workspace(op, args...; kw...)
        @test ws isa AK.Workspace
        @test AK._public_sizes(ws.sizes) == sizes
        @test occursin("Workspace(", sprint(show, ws))
        # The operation runs with it, as without it
        @test (op(args...; kw..., workspace=ws); true)
    end

    if TEST_KERNELS
        # Scratch mirrors the algorithm: radix sort's histograms and nested operations, none for
        # the bitonic network
        sizes = AK.workspace_size(AK.sort!, v; alg=AK.RadixSort())
        @test Base.all(k -> haskey(sizes, k), (:temp, :hist, :scan, :key_range))
        @test sizes.temp == (Float32, (10_000,))
        @test AK.workspace_size(AK.sort!, v; alg=AK.BitonicSort()) == (;)
        # A whole-array reduction keeps its partial results in two halves of one buffer
        @test haskey(AK.workspace_size(AK.sum, v; alg=AK.BlockReduce()), :partials)
        # ... which a reduction of one element does not need
        @test AK.workspace_size(AK.sum, v[1:1]; alg=AK.BlockReduce()) == (; partials=(Float32, (0,)))
        # DecoupledLookback's flags only where it runs
        if TEST_DL
            @test haskey(AK.workspace_size(AK.accumulate!, +, copy(v); alg=AK.DecoupledLookback()), :flags)
        end
        @test !haskey(AK.workspace_size(AK.accumulate!, +, copy(v); alg=AK.ScanPrefixes()), :flags)
        # `acctype` sets the partial results' type
        m = reshape(v, 10, 1000)
        R = array_from_host(zeros(Float32, 1, 1000))
        @test AK.workspace_size(AK.sum, v; alg=AK.BlockReduce(), acctype=Int32).partials[1] === Int32
        @test AK.workspace_size(AK.mapreducedim!, identity, +, array_from_host(zeros(Float32, 1, 1)),
                                reshape(v, 100, 100); alg=AK.BlockReduce(),
                                acctype=Int32).partials[1] === Int32
    end
    # A scan into a destination of its running type needs no scratch array; one into another needs
    # one of the running type, which the destination and `acctype` set
    w = array_from_host(rand(Int32(1):Int32(9), 1000))
    @test !haskey(AK.workspace_size(AK.accumulate!, +, similar(w, Int), w), :work)
    @test AK.workspace_size(AK.accumulate!, +, similar(w, Int8), w).work == (Int32, (1000,))
    @test AK.workspace_size(AK.accumulate!, +, similar(w), w; acctype=Int16).work ==
          (Int16, (1000,))
end


@testset "workspace: reuse and checks" begin
    Random.seed!(0)
    vh = rand(Float32, 10_000)
    v = array_from_host(vh)

    # One workspace serves any number of calls with the same plan
    ws = AK.workspace(AK.sort!, v)
    for _ in 1:3
        w = array_from_host(rand(Float32, 10_000))
        wh = Array(w)
        AK.sort!(w; workspace=ws)
        @test Array(w) == sort(wh)
    end
    ws = AK.workspace(AK.sum, v)
    @test AK.sum(v; workspace=ws) ≈ sum(vh)
    @test AK.sum(v; workspace=ws) ≈ sum(vh)
    # ... also by an empty `sum`, and not by one with another accumulator type
    e = array_from_host(zeros(Float32, 0, 3))
    @test Array(AK.sum(e; dims=1, workspace=AK.workspace(AK.sum, e; dims=1))) == zeros(1, 3)
    @test_throws ArgumentError AK.sum(e; dims=1, workspace=:invalid)
    @test_throws ArgumentError AK.prod(e[:, 1]; workspace=:invalid)
    if TEST_KERNELS
        @test_throws ArgumentError AK.sum(v; alg=AK.BlockReduce(),
                                          workspace=AK.workspace(AK.sum, v; acctype=Int32,
                                                                 alg=AK.BlockReduce()))
    end

    # ... and is rejected by a call with another plan: other buffer sizes, another algorithm,
    # another operation's buffers, or no `Workspace` at all
    alg = TEST_KERNELS ? AK.MergeSort() : AK.CPUThreads.SampleSort(max_tasks=4)
    @test_throws ArgumentError AK.sort!(array_from_host(rand(Float32, 20_000)); alg,
                                        workspace=AK.workspace(AK.sort!, v; alg))
    if TEST_KERNELS
        a = AK.BlockReduce()
        @test_throws ArgumentError AK.sum(array_from_host(rand(Float32, 50_000)); alg=a,
                                          workspace=AK.workspace(AK.sum, v; alg=a))
    end
    @test_throws ArgumentError AK.sort!(copy(v); workspace=AK.workspace(AK.sum, v))
    @test_throws ArgumentError AK.sort!(copy(v); workspace=similar(v))
    if TEST_KERNELS
        @test_throws ArgumentError AK.sort!(copy(v); alg=AK.MergeSort(),
                                            workspace=AK.workspace(AK.sort!, v; alg=AK.RadixSort()))
    end

    # A workspace must not alias the operation's arrays
    alg = TEST_KERNELS ? AK.MergeSort() : AK.CPUThreads.SampleSort(max_tasks=4)
    w = copy(v)
    ws = AK.workspace(AK.sort!, w; alg)
    @test haskey(ws.buffers, :temp)
    aliased = AK.Workspace(ws.backend, ws.device, ws.alg, ws.nested, ws.sizes,
                           merge(ws.buffers, (; temp=w)))
    @test_throws ArgumentError AK.sort!(w; alg, workspace=aliased)

    # The scratch keywords are gone
    @test_throws MethodError AK.sort!(copy(v); temp=similar(v))
    @test_throws MethodError AK.sum(v; temp=similar(v))
    @test_throws MethodError AK.accumulate!(+, copy(v); temp=similar(v))
    @test_throws MethodError AK.findall(x -> x > 0.5f0, v; temp_bools=similar(v, Bool))
end


@testset "workspace: aliasing, nested algorithms, inference" begin
    v = array_from_host(rand(Float32, 10_000))

    # An allocating sort checks the workspace against its input, not only against the copy
    alg = TEST_KERNELS ? AK.MergeSort() : AK.CPUThreads.SampleSort(max_tasks=4)
    ws = AK.workspace(AK.sort, v; alg)
    w = copy(v)
    aliased = AK.Workspace(ws.backend, ws.device, ws.alg, ws.nested, ws.sizes,
                           merge(ws.buffers, (; temp=w)))
    @test_throws ArgumentError AK.sort(w; alg, workspace=aliased)
    @test Array(AK.sort(w; alg, workspace=ws)) == sort(Array(w))

    if TEST_KERNELS
        # ... and a reduction against every array of a fused source
        x = array_from_host(ones(Int32, 1000))
        y = copy(x)
        a = AK.BlockReduce(block_size=2, items_per_thread=1)
        ws = AK.workspace(AK.mapreduce, +, +, x, y; alg=a)
        z = ws.buffers.partials
        @test length(z) == 1000
        @test_throws ArgumentError AK.mapreduce(+, +, z, y; alg=a, workspace=ws)
        # ... and a findall against its input
        ws = AK.workspace(AK.findall, identity, array_from_host(rand(Bool, 1000)))
        @test_throws ArgumentError AK.findall(identity, ws.buffers.mask; workspace=ws)

        # A `Broadcasted` source keeps its axes, on the host below `switch_below` too
        b = Base.Broadcast.broadcasted(identity, array_from_host(Int32[2]))
        bc = Base.Broadcast.Broadcasted(b.f, b.args, (Base.OneTo(1000),))
        for switch_below in (0, 2000)
            a = AK.BlockReduce(; switch_below)
            @test with_workspace(AK.mapreduce, identity, +, bc; alg=a) == 2000
        end

        # A fused source below `switch_below` finishes on the host without device scratch
        a = AK.BlockReduce(switch_below=2000)
        @test AK.workspace_size(AK.mapreduce, +, +, x, y; alg=a).partials[2] == (0,)
        @test with_workspace(AK.mapreduce, +, +, x, y; alg=a) == 2000

        # The nested operations' algorithms are part of the workspace
        bools = array_from_host(rand(Bool, 10_000))
        alg = AK.ScanScatter()
        ws = AK.workspace(AK.findall, bools; alg)
        @test haskey(ws.nested, :scan)
        other = AK.Workspace(ws.backend, ws.device, ws.alg,
                             merge(ws.nested, (; scan=AK.ScanPrefixes(16, 1))), ws.sizes, ws.buffers)
        @test_throws ArgumentError AK.findall(bools; alg, workspace=other)
        @test Array(AK.findall(bools; alg, workspace=ws)) == findall(Array(bools))
        ws = AK.workspace(AK.sort!, v; alg=AK.RadixSort())
        @test Base.all(k -> haskey(ws.nested, k), (:scan, :key_range))
    else
        # A workspace of reference elements, on the host
        ws = AK.workspace(AK.sort!, fill("a", 100); alg=AK.CPUThreads.SampleSort(max_tasks=4))
        @test occursin("Workspace(", sprint(show, ws))
    end

    # Results still infer
    V = typeof(v)
    @test only(Base.return_types(x -> AK.sum(x), (V,))) === Float32
    @test only(Base.return_types(x -> AK.reduce(+, x), (V,))) === Float32
    @test only(Base.return_types(x -> AK.sum(x; dims=1), (V,))) <: AbstractVector{Float32}
    @test only(Base.return_types(x -> AK.accumulate(+, x), (V,))) <: AbstractVector{Float32}
end


# With a workspace, an operation allocates no device memory (measured where the backend counts it)
if @isdefined(CUDACore) && BACKEND isa CUDACore.CUDABackend
    @testset "workspace: no device allocations" begin
        # (`CUDACore.@allocated`, as a function: the macro is not defined when this file is loaded
        # for other back-ends)
        function device_bytes(f)
            b0 = CUDACore.alloc_stats.alloc_bytes
            f()
            return CUDACore.alloc_stats.alloc_bytes - b0
        end
        v = array_from_host(rand(Float32, 100_000))
        ks = array_from_host(rand(Int32(1):Int32(9), 100_000))
        m = reshape(v, 100, 1000)
        for (op, args, kw) in (
                (AK.sort!, (copy(v),), (; alg=AK.MergeSort())),
                (AK.sort!, (copy(v),), (; alg=AK.RadixSort())),
                (AK.sort!, (copy(v),), (; alg=AK.MergeSort(), by=abs)),
                (AK.sortperm!, (similar(v, Int), v), (; alg=AK.MergeSort())),
                (AK.sortperm!, (similar(v, Int), v), (; alg=AK.MergeSort(lowmem=true))),
                (AK.sort_by_key!, (copy(ks), copy(v)), (; alg=AK.MergeSort())),
                (AK.sum, (v,), (;)),
                # (before Julia 1.12 a reduction of several arrays materializes them, see `Workspace`)
                (VERSION >= v"1.12-" ? ((AK.mapreduce, (*, +, v, v), (;)),) : ())...,
                (AK.mapreducedim!, (identity, +, similar(v, 1, 1000), m), (;)),
                (AK.mapreducedim!, (identity, +, similar(v, 100, 1), m), (;)),
                (AK.accumulate!, (+, copy(v)), (;)), (AK.accumulate!, (+, similar(v, Float64), v), (;)),
                (AK.any, (x -> x > 2, v), (; alg=AK.ConcurrentWrite())),
                (AK.any, (x -> x > 2, v), (; alg=AK.ViaReduce())),
            )
            ws = AK.workspace(op, args...; kw...)
            op(args...; kw..., workspace=ws)
            @test device_bytes(() -> op(args...; kw..., workspace=ws)) == 0
        end
    end
end
