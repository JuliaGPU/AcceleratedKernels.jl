# The workspace protocol: `workspace_size` and `workspace` plan the same buffers as the operation,
# which then allocates no scratch of its own; a workspace made for another call is rejected.

# The `(eltype, dims)` pairs of a workspace's buffers, as `workspace_size` gives them
buffer_sizes(buffers::NamedTuple) = map(buffer_sizes, buffers)
buffer_sizes(buffer::AbstractArray) = (eltype(buffer), size(buffer))

# `ws` with other buffers or another plan, which the operations must reject
forge(ws; buffers=ws.buffers, plan=ws.plan) = AK.Workspace(plan, ws.device, buffers)

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
            # Keywords that change the buffers' types
            (AK.sum, (v,), (; acctype=Float32)), (AK.reduce, (+, reshape(v, 100, 100)), (; dims=1, acctype=Float32)),
            (AK.mapreducedim!, (identity, +, array_from_host(zeros(Float32, 1, 100)), reshape(v, 100, 100)), (; acctype=Float32)),
            (AK.accumulate!, (+, similar(ks), ks), (; acctype=Int)), (AK.accumulate, (+, v), (; acctype=Float32)),
        )
        sizes = AK.workspace_size(op, args...; kw...)
        check_sizes(sizes)
        ws = AK.workspace(op, args...; kw...)
        @test ws isa AK.Workspace
        @test buffer_sizes(ws.buffers) == sizes
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
        # ... the same buffers, without elements, where the slices need no sorting or sort in one
        # block
        nelems(s) = s isa NamedTuple ? Base.sum(nelems, values(s); init=0) : Base.prod(s[2])
        short = AK.workspace_size(AK.sort!, v[1:1]; alg=AK.RadixSort())
        @test keys(short) == keys(sizes) && nelems(short) == 0
        @test AK.workspace_size(AK.sort!, fill!(similar(v, ()), 1); alg=AK.RadixSort()) == (;)
        for (h, dims) in ((rand(Float32, 0), :), (rand(Float32, 1), :), (rand(Float32, 100), :),
                          (rand(Float32, 10_000), :), (rand(Float32, 100, 3), 1),
                          (rand(Float32, 3, 3000), 2))
            @test Array(with_workspace(AK.sort!, array_from_host(h); alg=AK.RadixSort(), dims)) ==
                  (dims isa Colon ? sort(h) : sort(h; dims))
        end
        @test AK.workspace_size(AK.sort!, v; alg=AK.BitonicSort()) == (;)
        # Merge sort lists the same buffers for every length, empty where the call does not need
        # them: tiles that need no merging need no swap buffers, the low-memory permutation copies
        # no keys
        alg = AK.MergeSort(block_size=64)
        @test AK.workspace_size(AK.sort!, v[1:100]; alg) == (; temp=(Float32, (0,)))
        @test AK.workspace_size(AK.sortperm!, similar(v, Int, 100), v[1:100]; alg) ==
              (; keys=(Float32, (100,)), temp_keys=(Float32, (0,)), temp_values=(Int, (0,)))
        @test AK.workspace_size(AK.sortperm!, similar(v, Int), v; alg=AK.MergeSort(lowmem=true)) ==
              (; keys=(Float32, (0,)), temp_keys=(Float32, (0,)), temp_values=(Int, (10_000,)))
        # (of the array's shape along `dims`)
        @test AK.workspace_size(AK.sort!, reshape(v, 100, 100); alg, dims=1) ==
              (; temp=(Float32, (0, 0)))
        # (none for a zero-dimensional array, which has no empty buffers of its shape, or for the
        # keys of a low-memory permutation if they are of a union type, which backends may not
        # allocate)
        z = fill!(similar(v, ()), 1)
        @test AK.workspace_size(AK.sort!, z; alg) == (;)
        @test AK.workspace_size(AK.sortperm!, similar(v, Int, ()), z; alg) == (;)
        @test AK.workspace_size(AK.sort_by_key!, z, similar(v, Int, ()); alg) == (;)
        @test AK.workspace_size(AK.sortperm!, similar(v, Int, ()), v[1:1]; alg) == (;)
        @test AK.workspace_size(AK.sort_by_key!, v[1:1], similar(v, Int, ()); alg) == (;)
        by = x -> x < 0.5f0 ? Int16(1) : x
        @test AK.workspace_size(AK.sortperm!, similar(v, Int), v; alg=AK.MergeSort(lowmem=true),
                                by) == (; temp_values=(Int, (10_000,)))
        @test Array(with_workspace(AK.sortperm!, similar(v, Int), v; alg=AK.MergeSort(lowmem=true),
                                   by)) == sortperm(Array(v); by)
        # ... with which the calls run, of empty arrays too
        for n in (0, 1, 100, 1000)
            h = rand(Float32, n)
            hk = rand(Int32(1):Int32(9), n)
            @test Array(with_workspace(AK.sort!, array_from_host(h); alg)) == sort(h)
            @test Array(with_workspace(AK.sort!, array_from_host(h); alg, by=x -> -x)) ==
                  sort(h; rev=true)
            for lowmem in (false, true)
                x = array_from_host(h)
                a = AK.MergeSort(; block_size=64, lowmem)
                @test Array(with_workspace(AK.sortperm!, similar(x, Int), x; alg=a)) == sortperm(h)
            end
            ks, vs = with_workspace(AK.sort_by_key!, array_from_host(hk), array_from_host(1:n, Int);
                                    alg)
            @test Array(ks) == sort(hk) && Array(vs) == sortperm(hk)
        end
        # ... and along `dims`, with slices that need no merging and slices that do
        for (shape, dims) in (((100, 3), 1), ((3, 300), 2))
            h = rand(Float32, shape)
            hk = rand(Int32(1):Int32(9), shape)
            @test Array(with_workspace(AK.sort!, array_from_host(h); alg, dims)) == sort(h; dims)
            @test Array(with_workspace(AK.sort!, array_from_host(h); alg, dims, by=x -> -x)) ==
                  sort(h; dims, rev=true)
            for lowmem in (false, true)
                x = array_from_host(h)
                a = AK.MergeSort(; block_size=64, lowmem)
                @test Array(with_workspace(AK.sortperm!, similar(x, Int), x; alg=a, dims)) ==
                      sortperm(h; dims)
            end
            ks, vs = with_workspace(AK.sort_by_key!, array_from_host(hk),
                                    array_from_host(reshape(1:length(hk), shape), Int); alg, dims)
            @test Array(ks) == sort(hk; dims) && Array(vs) == sortperm(hk; dims)
        end
        # A whole-array reduction keeps the partial results of its first pass, at most
        # `target_blocks`, and a slot for the last one
        @test haskey(AK.workspace_size(AK.sum, v; alg=AK.BlockReduce()), :partials)
        @test AK.workspace_size(AK.sum, array_from_host(rand(Float32, 10^7));
                                alg=AK.BlockReduce()).partials[2] ==
              (AK.reduce_tuning(BACKEND, Float32).target_blocks + 1,)
        # So does a reduction along `dims` into a single output, without the slot
        @test AK.workspace_size(AK.mapreducedim!, identity, +,
                                array_from_host(zeros(Float32, 1, 1)),
                                array_from_host(rand(Float32, 1000, 10_000));
                                alg=AK.BlockReduce()).partials[2] ==
              (AK.reduce_tuning(BACKEND, Float32).target_blocks,)
        # So does one split across blocks, also in element order
        x = array_from_host(rand(Float32, 100_000, 4))
        for commutative in (true, false)
            @test AK.workspace_size(AK.sum, x; dims=1, alg=AK.BlockReduce(),
                                    commutative).partials[2][1] == 4
        end
        # A reduction of one element needs none
        @test AK.workspace_size(AK.sum, v[1:1]; alg=AK.BlockReduce()) == (; partials=(Float32, (0,)))
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
    # Also by an empty `sum`, but not by one with another accumulator type
    e = array_from_host(zeros(Float32, 0, 3))
    @test Array(AK.sum(e; dims=1, workspace=AK.workspace(AK.sum, e; dims=1))) == zeros(1, 3)
    @test_throws ArgumentError AK.sum(e; dims=1, workspace=:invalid)
    @test_throws ArgumentError AK.prod(e[:, 1]; workspace=:invalid)
    if TEST_KERNELS
        @test_throws ArgumentError AK.sum(v; alg=AK.BlockReduce(),
                                          workspace=AK.workspace(AK.sum, v; acctype=Int32,
                                                                 alg=AK.BlockReduce()))
    end

    # Calls with another plan reject it: other buffer sizes, another algorithm, another
    # operation's buffers, or no `Workspace` at all
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
    aliased = forge(ws; buffers=(; ws.buffers..., temp=w))
    @test_throws ArgumentError AK.sort!(w; alg, workspace=aliased)

    # Operations without a workspace say so
    @test_throws ArgumentError AK.workspace_size(AK.map!, identity, similar(v), v)
    @test_throws ArgumentError AK.workspace(AK.foreachindex, identity, v)
end


@testset "workspace: aliasing, nested algorithms, inference" begin
    v = array_from_host(rand(Float32, 10_000))

    # An allocating sort checks the workspace against its input, not only against the copy
    alg = TEST_KERNELS ? AK.MergeSort() : AK.CPUThreads.SampleSort(max_tasks=4)
    ws = AK.workspace(AK.sort, v; alg)
    w = copy(v)
    aliased = forge(ws; buffers=(; ws.buffers..., temp=w))
    @test_throws ArgumentError AK.sort(w; alg, workspace=aliased)
    @test Array(AK.sort(w; alg, workspace=ws)) == sort(Array(w))

    if TEST_KERNELS
        # A reduction checks it against every array of a fused source
        x = array_from_host(ones(Int32, 1000))
        y = copy(x)
        # Two elements in one tile: the partials are the first pass's one and the result's
        a = AK.BlockReduce(block_size=2, items_per_thread=1)
        ws = AK.workspace(AK.mapreduce, +, +, x[1:2], y[1:2]; alg=a)
        z = ws.buffers.partials
        @test length(z) == 2
        @test_throws ArgumentError AK.mapreduce(+, +, z, y[1:2]; alg=a, workspace=ws)
        # findall checks it against its input
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
        p = ws.plan
        @test p.sizes.scan isa AK.Plan
        scan = AK.Plan(p.backend, AK.ScanPrefixes(16, 1), p.sizes.scan.sizes)
        other = forge(ws; plan=AK.Plan(p.backend, p.alg, (; p.sizes..., scan)))
        err = try AK.findall(bools; alg, workspace=other) catch e; e end
        @test err isa ArgumentError && occursin("ScanPrefixes(16, 1)", err.msg)
        @test Array(AK.findall(bools; alg, workspace=ws)) == findall(Array(bools))
        ws = AK.workspace(AK.sort!, v; alg=AK.RadixSort())
        @test Base.all(k -> ws.plan.sizes[k] isa AK.Plan, (:scan, :key_range))
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
    # ... and so do merge sort's plans, whatever the length and `lowmem`, and its workspaces where
    # the backend's allocations infer (OpenCL.jl chooses their memory at run time)
    infers(f, types...) = isconcretetype(only(Base.return_types(f, types)))
    I = typeof(similar(v, Int))
    M = typeof(reshape(v, 100, 100))
    allocates = infers(b -> KernelAbstractions.allocate(b, Float32, (1,)), typeof(BACKEND))
    for g in (allocates ? (AK.plan, AK.workspace) : (AK.plan,))
        # (as are those of `Auto()` for a permutation or a key/value sort, on every backend)
        @test infers((ix, x) -> g(AK.sortperm!, ix, x), I, V)
        @test infers((k, x) -> g(AK.sort_by_key!, k, x), V, I)
        TEST_KERNELS || continue
        @test infers(x -> g(AK.sort!, x; alg=AK.MergeSort()), V)
        @test infers(x -> g(AK.sort!, x; alg=AK.MergeSort(), by=abs), V)
        @test infers((ix, x, lowmem) -> g(AK.sortperm!, ix, x; alg=AK.MergeSort(; lowmem)),
                     I, V, Bool)
        @test infers((ix, x, lowmem) -> g(AK.sortperm!, ix, x; alg=AK.MergeSort(; lowmem), by=abs),
                     I, V, Bool)
        @test infers((x, lowmem) -> g(AK.sortperm, x; alg=AK.MergeSort(; lowmem)), V, Bool)
        @test infers((k, x) -> g(AK.sort_by_key!, k, x; alg=AK.MergeSort()), V, I)
        @test infers((x, dims) -> g(AK.sort!, x; alg=AK.MergeSort(), dims), M, Int)
        @test infers((x, dims, lowmem) -> g(AK.sortperm, x; alg=AK.MergeSort(; lowmem), dims),
                     M, Int, Bool)
        # ... as do radix sort's
        @test infers(x -> g(AK.sort!, x; alg=AK.RadixSort()), V)
        @test infers((x, dims) -> g(AK.sort!, x; alg=AK.RadixSort(), dims), M, Int)
    end
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
                (AK.mapreducedim!, (identity, +, similar(v, 1, 10), reshape(v, 10_000, 10)),
                 (; commutative=false)),
                (AK.mapreducedim!, (identity, +, similar(v, 100, 1), m), (; commutative=false)),
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
