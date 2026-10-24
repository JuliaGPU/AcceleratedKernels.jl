# Resolution of algorithms and of the backend, without launching kernels: resolution only looks at
# types, sizes, `dims` and orderings, so host arrays stand in for device arrays throughout.

# A GPU backend whose tunings the tests set with `with_tuning`, and arrays that live on it
struct FakeBackend <: KernelAbstractions.Backend end
const FAKE_TUNINGS = (sort=Ref(AK.SortTuning()), reduce=Ref(AK.ReduceTuning()),
                      scan=Ref(AK.ScanTuning()), findall=Ref(AK.FindallTuning()),
                      predicate=Ref(AK.PredicateTuning()))
AK.sort_tuning(::FakeBackend, ::Type) = FAKE_TUNINGS.sort[]
AK.reduce_tuning(::FakeBackend, ::Type) = FAKE_TUNINGS.reduce[]
AK.scan_tuning(::FakeBackend, ::Type) = FAKE_TUNINGS.scan[]
AK.findall_tuning(::FakeBackend, ::Type) = FAKE_TUNINGS.findall[]
AK.predicate_tuning(::FakeBackend, ::Type) = FAKE_TUNINGS.predicate[]

# Run `f` with `tuning` as the fake backend's tuning of its family (the others keep their defaults)
function with_tuning(f, tuning)
    ref = only(r for r in FAKE_TUNINGS if r[] isa typeof(tuning))
    old = ref[]
    ref[] = tuning
    try
        f()
    finally
        ref[] = old
    end
end

struct FakeArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::FakeArray) = size(a.data)
Base.getindex(a::FakeArray, i::Int...) = a.data[i...]
KernelAbstractions.get_backend(::FakeArray) = FakeBackend()

# A backend that cannot run AK's kernels, like KernelAbstractions 0.9's `CPU`
struct NoKernelsBackend <: KernelAbstractions.Backend end
AK.runs_kernels(::NoKernelsBackend) = false

const FAKE = FakeBackend()
const HOST = AK.HOST_BACKEND

# An algorithm that is not a sorting algorithm
struct NotASortAlgorithm <: AK.Algorithm end

# An array type that does not implement `get_backend`
struct NoBackendArray <: AbstractVector{Int} end
Base.size(::NoBackendArray) = (3,)
Base.getindex(::NoBackendArray, i::Int) = i

# A host vector that must not be asked for its backend
struct NoVoteVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(a::NoVoteVector) = size(a.data)
Base.getindex(a::NoVoteVector, i::Int) = a.data[i]
Base.setindex!(a::NoVoteVector, x, i::Int) = (a.data[i] = x)
KernelAbstractions.get_backend(::NoVoteVector) = error("backend queried despite an explicit backend")

const FWD = Base.Order.Forward
const REV = Base.Order.Reverse

resolve(alg, v; dims=:, ord=FWD, backend=FAKE, kw...) = AK.resolve_sort(alg, backend, v, dims, ord; kw...)


@testset "sort resolution: Auto" begin
    v = zeros(Float32, 1000)

    # The default tuning reproduces AK's historical choices and settings
    @test resolve(AK.Auto(), v) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), v; perm=true) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), v; pairs=true) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), zeros(Int32, 10^8 ÷ 100)) isa AK.MergeSort
    @test resolve(AK.Auto(), zeros(Int32, 10, 10); dims=1) isa AK.MergeSort

    # On the host backend: the threaded sample sort, filled
    a = resolve(AK.Auto(), v; backend=HOST)
    @test a isa AK.CPUThreads.SampleSort
    @test a.max_tasks == Threads.nthreads() && a.min_elems == 1
    @test resolve(AK.Auto(), v; backend=HOST, perm=true) isa AK.CPUThreads.SampleSort
    @test resolve(AK.Auto(), v; backend=HOST, pairs=true) isa AK.CPUThreads.SampleSort

    # Thresholds: bitonic sort up to and including `bitonic_max_len`, radix sort from
    # `radix_min_len`, for element types and orderings where they give the same result
    with_tuning(AK.SortTuning(bitonic_max_len=1024, radix_min_len=4096, radix_block_size=128,
                              radix_items_per_thread=4, bitonic_block_size=64, bitonic_items_per_thread=4)) do
        @test resolve(AK.Auto(), zeros(Int32, 1024)) === AK.BitonicSort(64, 4)
        @test resolve(AK.Auto(), zeros(Int32, 1025)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4095)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4096)) === AK.RadixSort(128, 4, 32)
        @test resolve(AK.Auto(), zeros(Int32, 4096); ord=REV) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 64, 16); dims=1) isa AK.BitonicSort
        @test resolve(AK.Auto(), zeros(Int32, 2048, 16); dims=1) isa AK.MergeSort

        # Radix sort is for whole arrays only: a vector or an N×1 matrix sorted along `dims`
        # is a slice, however long
        @test resolve(AK.Auto(), zeros(Int32, 10_000); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000, 1); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000, 1)) isa AK.RadixSort

        # Radix sort needs a supported element type and the default ordering or its reverse
        @test resolve(AK.Auto(), zeros(Int16, 10_000)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000); ord=Base.Order.ord(isless, abs, nothing)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000); ord=Base.Order.ord(>, identity, nothing)) isa AK.MergeSort

        # Stable by default: bitonic sort only where equal elements are bitwise identical
        for T in (Int8, UInt16, Int32, UInt64, Int128, Bool, Char)
            @test resolve(AK.Auto(), Vector{T}(undef, 100)) isa AK.BitonicSort
            @test resolve(AK.Auto(), Vector{T}(undef, 100); ord=REV) isa AK.BitonicSort
        end
        for T in (Float16, Float32, Float64, Tuple{Int32, Int32})
            @test resolve(AK.Auto(), Vector{T}(undef, 100)) isa AK.MergeSort
            @test resolve(AK.Auto(stable=false), Vector{T}(undef, 100)) isa AK.BitonicSort
        end
        @test resolve(AK.Auto(), Vector{Union{Int32, UInt32}}(undef, 100)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 100); ord=Base.Order.ord(isless, abs, nothing)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 100); ord=Base.Order.ord((a, b) -> a > b, identity, nothing)) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Int32, 100); ord=Base.Order.ord(isless, abs, nothing)) isa AK.BitonicSort

        # sortperm! and key/value sorting always observe stability, and need a permutation path
        @test resolve(AK.Auto(), zeros(Int32, 100); perm=true) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Float32, 100); perm=true) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Int32, 10_000); pairs=true) isa AK.MergeSort
    end

    # A different tuning changes the selection
    with_tuning(AK.SortTuning(radix_min_len=0)) do
        @test resolve(AK.Auto(), zeros(Float32, 10)) isa AK.RadixSort
    end
    @test resolve(AK.Auto(), zeros(Float32, 10)) isa AK.MergeSort

    # Selection never depends on the data
    @test resolve(AK.Auto(), fill(NaN32, 100)) === resolve(AK.Auto(), zeros(Float32, 100))
end


@testset "sort resolution: explicit algorithms" begin
    v = zeros(Float32, 1000)

    # Explicit fields win over the tuning, unset ones come from it
    with_tuning(AK.SortTuning(merge_block_size=128, radix_block_size=512, radix_items_per_thread=8,
                              radix_chunk_size=64, bitonic_block_size=32, bitonic_items_per_thread=16,
                              threads_min_elems=7)) do
        @test resolve(AK.MergeSort(), v) === AK.MergeSort(128, false)
        @test resolve(AK.MergeSort(block_size=64), v) === AK.MergeSort(64, false)
        @test resolve(AK.MergeSort(lowmem=true), v; perm=true) === AK.MergeSort(128, true)
        @test resolve(AK.RadixSort(), v) === AK.RadixSort(512, 8, 64)
        @test resolve(AK.RadixSort(items_per_thread=1), v) === AK.RadixSort(512, 1, 64)
        @test resolve(AK.RadixSort(chunk_size=16), v) === AK.RadixSort(512, 8, 16)
        @test resolve(AK.BitonicSort(block_size=128), v) === AK.BitonicSort(128, 16)
    end
    @test resolve(AK.CPUThreads.SampleSort(max_tasks=3), v; backend=HOST) ===
          AK.CPUThreads.SampleSort(3, 1)
    @test resolve(AK.CPUThreads.SampleSort(min_elems=5), v; backend=HOST) ===
          AK.CPUThreads.SampleSort(Threads.nthreads(), 5)

    # Rejections, per algorithm (and none for valid combinations)
    rejects(alg, v=v; kw...) = try
        resolve(alg, v; kw...)
        false
    catch err
        err isa ArgumentError || rethrow()
        true
    end

    # MergeSort
    @test rejects(AK.MergeSort(block_size=0))
    @test !rejects(AK.MergeSort(block_size=100))                    # any positive size
    @test rejects(AK.MergeSort(lowmem=true))                        # sort!
    @test rejects(AK.MergeSort(lowmem=true); pairs=true)            # sort_by_key!
    @test !rejects(AK.MergeSort(lowmem=true); perm=true)
    @test !rejects(AK.MergeSort(), zeros(Float32, 4, 4); dims=2, perm=true)

    # RadixSort
    @test rejects(AK.RadixSort(block_size=0))
    @test rejects(AK.RadixSort(block_size=100))
    @test rejects(AK.RadixSort(items_per_thread=0))
    @test rejects(AK.RadixSort(); perm=true)
    @test rejects(AK.RadixSort(); pairs=true)
    @test rejects(AK.RadixSort(), zeros(Int16, 16))
    @test rejects(AK.RadixSort(), Vector{Tuple{Int32, Int32}}(undef, 16))
    @test rejects(AK.RadixSort(); ord=Base.Order.ord(isless, abs, nothing))
    @test rejects(AK.RadixSort(); ord=Base.Order.ord(>, identity, nothing))
    @test rejects(AK.RadixSort(block_size=8192))                    # local memory
    @test rejects(AK.RadixSort(block_size=1 << 62))                  # no overflowing footprints
    @test rejects(AK.RadixSort(items_per_thread=1 << 62))
    @test rejects(AK.RadixSort(items_per_thread=65))
    @test rejects(AK.RadixSort(chunk_size=0))
    @test rejects(AK.RadixSort(chunk_size=24))
    @test rejects(AK.RadixSort(chunk_size=2048))
    if Sys.WORD_SIZE == 64                                              # positions are UInt32s
        @test !rejects(AK.RadixSort(), Base.OneTo(2^32))
        @test rejects(AK.RadixSort(), Base.OneTo(2^32 + 1))
        @test !rejects(AK.RadixSort(), reshape(Base.OneTo(2^33 + 2), 2, :); dims=1)
        @test rejects(AK.RadixSort(), reshape(Base.OneTo(2^33 + 2), :, 2); dims=1)
    end
    for T in (UInt32, Int32, Float32, UInt64, Int64, Float64), ord in (FWD, REV)
        @test !rejects(AK.RadixSort(), zeros(T, 16); ord)
    end
    @test !rejects(AK.RadixSort(), zeros(Float32, 4, 4))             # dims=: sorts flat
    @test !rejects(AK.RadixSort(), zeros(Float32, 4, 4); dims=1)
    @test !rejects(AK.RadixSort(), zeros(Float32, 4, 4); dims=2)
    @test !rejects(AK.RadixSort(), zeros(Float32, 16); dims=1)

    # BitonicSort
    @test rejects(AK.BitonicSort(block_size=100))
    @test rejects(AK.BitonicSort(items_per_thread=3))
    @test rejects(AK.BitonicSort(block_size=2, items_per_thread=1 << (Sys.WORD_SIZE - 2)))
    @test rejects(AK.BitonicSort(); perm=true)
    @test rejects(AK.BitonicSort(); pairs=true)
    @test !rejects(AK.BitonicSort(), zeros(Float32, 4, 4); dims=2, ord=Base.Order.ord(isless, abs, nothing))

    # CPUThreads.SampleSort only on the host, where every operation supports it
    @test rejects(AK.CPUThreads.SampleSort())
    @test rejects(AK.CPUThreads.SampleSort(max_tasks=0); backend=HOST)
    @test rejects(AK.CPUThreads.SampleSort(min_elems=0); backend=HOST)
    @test !rejects(AK.CPUThreads.SampleSort(); backend=HOST, perm=true)
    @test !rejects(AK.CPUThreads.SampleSort(); backend=HOST, pairs=true, dims=1)

    # Kernel algorithms need a backend that runs AK's kernels
    for alg in (AK.MergeSort(), AK.RadixSort(), AK.BitonicSort())
        @test rejects(alg; backend=NoKernelsBackend())
        @test rejects(alg; backend=HOST) == !AK.runs_kernels(HOST)
    end

    # Domain checks run before anything else, whatever the input length
    @test rejects(AK.MergeSort(block_size=-1), zeros(Float32, 0))
    @test rejects(AK.RadixSort(block_size=3), zeros(Float32, 1))

    # Invalid `dims`
    @test rejects(AK.Auto(), zeros(Float32, 4, 4); dims=3)
    @test rejects(AK.Auto(), zeros(Float32, 4, 4); dims=0)

    # Not a sorting algorithm
    @test rejects(NotASortAlgorithm())
end


@testset "sort resolution: inference" begin
    for T in (Int32, Float32), (A, D) in ((Vector{T}, Colon), (Matrix{T}, Int))
        # Auto resolves to a small union of concrete algorithms, not `Any`
        rt = only(Base.return_types(AK.resolve_sort,
                                    (AK.Auto, FakeBackend, A, D, Base.Order.ForwardOrdering)))
        @test rt <: Union{AK.MergeSort, AK.RadixSort, AK.BitonicSort}
        @test rt isa Union || isconcretetype(rt)
        rt = only(Base.return_types(AK.resolve_sort,
                                    (AK.Auto, typeof(HOST), A, D, Base.Order.ForwardOrdering)))
        @test rt === AK.CPUThreads.SampleSort
        # An explicit algorithm resolves to itself
        rt = only(Base.return_types(AK.resolve_sort,
                                    (AK.RadixSort, FakeBackend, A, D, Base.Order.ForwardOrdering)))
        @test rt === AK.RadixSort
    end
end


@testset "backend resolution" begin
    d = FakeArray(zeros(Float32, 4))
    h = zeros(Float32, 4)

    # Every array votes, destination first; backend-free leaves do not
    @test AK.resolve_backend(nothing, d) === FAKE
    @test AK.resolve_backend(nothing, d, 1:4, CartesianIndices((2, 2)), LinearIndices(h),
                             1.0, Ref(2), (3, 1:2), nothing) === FAKE
    @test AK.resolve_backend(nothing, 1:4, d) === FAKE
    @test AK.resolve_backend(nothing, h, 1:4) == HOST
    # ... nor do Base's views, reshapes and permutations of them, while those of arrays vote
    @test AK.resolve_backend(nothing, d, reshape(1:6, 2, 3), view(1:10, 2:4),
                             PermutedDimsArray(reshape(1:6, 2, 3), (2, 1))) === FAKE
    @test AK.resolve_backend(nothing, view(d, 1:2)) === FAKE
    @test AK.resolve_backend(nothing, reshape(1:6, 2, 3)) == HOST

    # All votes must agree, including the non-destination inputs
    @test_throws ArgumentError AK.resolve_backend(nothing, d, h)
    @test_throws ArgumentError AK.resolve_backend(nothing, h, d)
    @test_throws ArgumentError AK.resolve_backend(nothing, h, (1, d))
    # ... unless the backend is explicit
    @test AK.resolve_backend(FAKE, h) === FAKE
    @test AK.resolve_backend(HOST, d, h) == HOST

    # Broadcasted trees are walked
    bc = Base.Broadcast.broadcasted(+, d, Base.Broadcast.broadcasted(*, 2, 1:4))
    @test AK.resolve_backend(nothing, bc) === FAKE
    @test_throws ArgumentError AK.resolve_backend(nothing, h, bc)
    # ... including the arrays of a preprocessed one
    pbc = Base.Broadcast.preprocess(nothing,
        Base.Broadcast.instantiate(Base.Broadcast.broadcasted(identity, d)))
    @test AK.resolve_backend(nothing, pbc) === FAKE
    @test_throws ArgumentError AK.resolve_backend(nothing, h, pbc)

    # Nothing votes: the host backend
    @test AK.resolve_backend(nothing) == HOST
    @test AK.resolve_backend(nothing, 1:10, 2.0) == HOST

    # Array types without `get_backend` keep raising KernelAbstractions' error
    @test_throws ArgumentError AK.resolve_backend(nothing, NoBackendArray())
    @test AK.resolve_backend(FAKE, NoBackendArray()) === FAKE
    # ... and allocating operations, given the backend, do not ask for it either
    @test AK.sortperm(NoBackendArray(); backend=HOST) == [1, 2, 3]
    @test AK.backendless(reshape(1:6, 2, 3)) && !AK.backendless(NoBackendArray())

    # The sorting entry points resolve from all their arrays, and accept only backends
    @test_throws MethodError AK.sort!(d; temp=h)      # no scratch keywords: see `workspace`
    @test_throws ArgumentError AK.sortperm!(zeros(Int, 4), d)
    @test_throws ArgumentError AK.sort_by_key!(d, h)
    @test_throws TypeError AK.sort!(h; backend=:cpu)
    # ... and never take a positional backend
    @test_throws MethodError AK.sort!(h, HOST)

    # Nested operations receive the resolved backend instead of querying the arrays again
    alg = AK.CPUThreads.SampleSort(max_tasks=4)
    v = NoVoteVector(Int32[3, 1, 2])
    @test AK.sort!(v; backend=HOST, alg).data == [1, 2, 3]
    ix = NoVoteVector(zeros(Int, 100))
    vals = rand(Int32, 100)
    @test AK.sortperm!(ix, vals; backend=HOST, alg).data == sortperm(vals)
    k = NoVoteVector(rand(Int32(1):Int32(3), 100))
    kv = copy(k.data)
    AK.sort_by_key!(k, NoVoteVector(collect(1:100)); backend=HOST, alg)
    @test k.data == sort(kv)
end


resolve_reduce(alg; T=Float32, dims=:, backend=FAKE) = AK.resolve_reduce(alg, backend, T, dims)


@testset "reduce resolution: Auto" begin
    # The default tuning, for whole arrays and `dims`
    @test resolve_reduce(AK.Auto()) === AK.BlockReduce(256, 16, 0)
    @test resolve_reduce(AK.Auto(); dims=nothing) === AK.BlockReduce(256, 16, 0)
    @test resolve_reduce(AK.Auto(); dims=2) === AK.BlockReduce(256, 16, 0)
    @test resolve_reduce(AK.Auto(); dims=(1, 3)) === AK.BlockReduce(256, 16, 0)

    # On the host backend: the threaded algorithm, filled
    @test resolve_reduce(AK.Auto(); backend=HOST) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    @test resolve_reduce(AK.Auto(); backend=HOST, dims=1) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)

    # The tuning fills every field; `stable` means nothing to a reduction
    with_tuning(AK.ReduceTuning(block_size=512, items_per_thread=4, switch_below=100)) do
        @test resolve_reduce(AK.Auto()) === AK.BlockReduce(512, 4, 100)
        @test resolve_reduce(AK.Auto(stable=false)) === AK.BlockReduce(512, 4, 100)
    end

    # A tuning without `items_per_thread` loads 64 bytes of accumulators per thread and tile, 4 to
    # 16 elements
    @test resolve_reduce(AK.Auto(); T=Float64) === AK.BlockReduce(256, 8, 0)
    @test resolve_reduce(AK.Auto(); T=NTuple{8, Float64}) === AK.BlockReduce(256, 4, 0)
    @test resolve_reduce(AK.Auto(); T=Union{}) === AK.BlockReduce(256, 4, 0)
    with_tuning(AK.ReduceTuning(block_size=512, switch_below=100)) do
        @test resolve_reduce(AK.Auto(); T=Int8) === AK.BlockReduce(512, 16, 100)
    end
end


@testset "reduce resolution: explicit algorithms" begin
    # Explicit fields win over the tuning, unset ones come from it
    with_tuning(AK.ReduceTuning(block_size=512, items_per_thread=4, switch_below=100,
                                threads_min_elems=7)) do
        @test resolve_reduce(AK.BlockReduce()) === AK.BlockReduce(512, 4, 100)
        @test resolve_reduce(AK.BlockReduce(block_size=64)) === AK.BlockReduce(64, 4, 100)
        @test resolve_reduce(AK.BlockReduce(items_per_thread=1, switch_below=0)) ===
              AK.BlockReduce(512, 1, 0)
        @test resolve_reduce(AK.BlockReduce(block_size=128); dims=1) === AK.BlockReduce(128, 4, 100)
    end
    @test resolve_reduce(AK.CPUThreads.Partitioned(max_tasks=3); backend=HOST) ===
          AK.CPUThreads.Partitioned(3, 1)
    @test resolve_reduce(AK.CPUThreads.Partitioned(min_elems=1000); backend=HOST) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1000)

    # Domains of explicit fields
    for bad in (AK.BlockReduce(block_size=0), AK.BlockReduce(block_size=192),
                AK.BlockReduce(block_size=2048), AK.BlockReduce(items_per_thread=0),
                AK.BlockReduce(switch_below=-1))
        @test_throws ArgumentError resolve_reduce(bad)
    end
    for bad in (AK.CPUThreads.Partitioned(max_tasks=0), AK.CPUThreads.Partitioned(min_elems=0))
        @test_throws ArgumentError resolve_reduce(bad; backend=HOST)
    end

    # `items_per_thread` and `switch_below` only apply to whole-array reductions; they used to be
    # ignored silently along `dims`
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=1)
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(switch_below=0); dims=(1, 2))
    @test resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=nothing) isa AK.BlockReduce

    # Whole-array reductions need tiles of at least two elements (one never shrinks the input)
    # and at most typemax(Int32), whether the settings are explicit or from the tuning
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=1))
    @test resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=2)) === AK.BlockReduce(1, 2, 0)
    @test resolve_reduce(AK.BlockReduce(block_size=1); dims=1) === AK.BlockReduce(1, 16, 0)
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=1 << 56))
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1024, items_per_thread=1 << 22))
    with_tuning(AK.ReduceTuning(block_size=1, items_per_thread=1)) do
        @test_throws ArgumentError resolve_reduce(AK.Auto())
        @test resolve_reduce(AK.Auto(); dims=2) === AK.BlockReduce(1, 1, 0)
    end

    # Capabilities: the threaded algorithm only on the host, kernels only where they run
    @test_throws ArgumentError resolve_reduce(AK.CPUThreads.Partitioned())
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); backend=NoKernelsBackend())
    if AK.runs_kernels(HOST)
        @test resolve_reduce(AK.BlockReduce(); backend=HOST) === AK.BlockReduce(256, 16, 0)
    else
        @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); backend=HOST)
    end

    # Algorithms of other families
    @test_throws ArgumentError resolve_reduce(AK.MergeSort())
    @test_throws ArgumentError resolve_reduce(AK.CPUThreads.SampleSort(); backend=HOST)
    @test_throws ArgumentError AK.resolve_sort(AK.BlockReduce(), FAKE, zeros(Float32, 10), :,
                                               Base.Order.Forward; perm=false)
end


@testset "reduce resolution: tuning values" begin
    # A tuning cannot make a reduction launch nothing
    with_tuning(AK.ReduceTuning(target_blocks=0)) do
        @test_throws ArgumentError resolve_reduce(AK.BlockReduce())
        @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); dims=1)
    end
end


@testset "reduce resolution: inference" begin
    for (B, dims) in ((FakeBackend, Colon), (FakeBackend, Int),
                      (typeof(HOST), Nothing), (typeof(HOST), Tuple{Int, Int}))
        rt = only(Base.return_types(AK.resolve_reduce, (AK.Auto, B, Type{Float32}, dims)))
        @test rt <: Union{AK.BlockReduce, AK.CPUThreads.Partitioned}
    end
end


@testset "reduce resolution: entry points" begin
    v = rand(Int32(1):Int32(100), 1000)
    m = rand(Int32(1):Int32(100), 10, 100)

    # Invalid explicit algorithms are rejected before any data is touched, whatever the length
    for x in (v, Int32[], m)
        @test_throws ArgumentError AK.reduce(+, x; init=Int32(0), alg=AK.BlockReduce(block_size=3))
        @test_throws ArgumentError AK.mapreduce(abs, +, x; init=Int32(0), alg=AK.MergeSort())
        @test_throws ArgumentError AK.sum(x; alg=AK.CPUThreads.Partitioned(max_tasks=0))
    end
    @test_throws ArgumentError AK.reduce(+, m; init=Int32(0), dims=1,
                                         alg=AK.BlockReduce(items_per_thread=4))

    # The convenience reductions forward `alg`
    alg = AK.CPUThreads.Partitioned(max_tasks=4, min_elems=10)
    @test AK.sum(v; alg) == sum(v)
    @test AK.prod(Int64.(v[1:5]); alg) == prod(Int64.(v[1:5]))
    @test AK.maximum(v; alg) == maximum(v)
    @test AK.minimum(v; alg) == minimum(v)
    @test AK.count(>(50), v; alg) == count(>(50), v)
    @test AK.sum(m; dims=2, alg) == sum(m; dims=2)

    # The loose settings are gone
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), block_size=256)
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), max_tasks=2)
    @test_throws MethodError AK.reduce(+, v; init=Int32(0), prefer_threads=false)
    @test_throws MethodError AK.sum(v; switch_below=10)
end


@testset "reduce resolution: accumulator and seed" begin
    # The accumulator type follows Base's promotion to a fixed point
    @test AK.accumulator_type(Base.add_sum, Int8, Int8) === Int
    @test AK.accumulator_type(+, Int8, Int8) === Int8
    @test AK.accumulator_type(+, Bool, Bool) === Int
    @test AK.accumulator_type(+, Int, Float32) === Float32
    @test AK.accumulator_type(max, Tuple{Int32, Int32}, Tuple{Int32, Int32}) === Tuple{Int32, Int32}
    @test AK.first_type(Base.add_sum, Int8) === Int
    if VERSION >= v"1.13-"      # older `add_sum`s keep mixed-signedness small integers small
        @test AK.accumulator_type(Base.add_sum, UInt8, Int8) === Int
        @test AK.accumulator_type(Base.add_sum, UInt8, Bool) === Int
    end
    @test AK.accumulator_type(+, Union{}, Char) === Union{}
    @test AK.accumulator_type(Returns(false), Union{}, Int) === Int
    @test AK.accumulator_type(coalesce, Int, Union{Missing, Int}) === Union{Missing, Int}
    # `init` is applied once, not a partial result: only `op(init, x)`'s type counts
    @test AK.accumulator_type((a, b) -> something(a, 0) + something(b, 0), Nothing, Int) === Int

    # Seeds: the caller's neutral, else GPUArraysCore's, else an empty lane
    @test AK.reduce_seed(+, Int32, nothing) === Int32(0)
    @test AK.reduce_seed(min, Float32, nothing) === Inf32
    @test AK.reduce_seed(+, Float32, nothing) === -0.0f0
    @test AK.reduce_seed(Base.add_sum, ComplexF32, nothing) === complex(-0.0f0, -0.0f0)
    @test AK.reduce_seed(+, Int32, 0) === Int32(0)
    @test AK.reduce_seed((a, b) -> a + b, Int32, nothing) isa AK.Optional{Int32}
    @test !AK.isset(AK.reduce_seed((a, b) -> a + b, Int32, nothing))
    # Empty lanes of every kind of type
    @test !AK.isset(AK.Optional{Tuple{Int8, Int64}}())
    @test !AK.isset(AK.Optional{Union{Missing, Int}}())
    @test !AK.isset(AK.Optional{String}())
    @test AK.isset(AK.Optional{String}("x"))

    # Results infer, whether or not the operator has a known neutral element
    @test only(Base.return_types(v -> AK.sum(v), (Vector{Int8},))) === Int
    @test only(Base.return_types(v -> AK.reduce((a, b) -> a + b, v), (Vector{Int32},))) === Int32
    @test only(Base.return_types(m -> AK.sum(m; dims=1), (Matrix{Int8},))) === Matrix{Int}
    @test only(Base.return_types(v -> AK.mapreduce(tuple, (a, b) -> a, v, v),
                                 (Vector{Int32},))) === Tuple{Int32, Int32}

    # BlockReduce needs a bits-type accumulator; the host algorithm does not
    @test_throws ArgumentError resolve_reduce(AK.Auto(); T=Union{Missing, Int32})
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); T=String)
    @test resolve_reduce(AK.Auto(); T=String, backend=HOST) isa AK.CPUThreads.Partitioned
end


resolve_scan(alg; T=Float32, dims=nothing, backend=FAKE) = AK.resolve_scan(alg, backend, T, dims)


@testset "scan resolution: Auto" begin
    # The default tuning reproduces AK's historical settings: 256 threads, at most 8 items per
    # thread, fewer for wide element types
    @test resolve_scan(AK.Auto()) === AK.ScanPrefixes(256, 8)
    @test resolve_scan(AK.Auto(); T=ComplexF64) === AK.ScanPrefixes(256, 7)
    @test resolve_scan(AK.Auto(); T=NTuple{16, Float64}) === AK.ScanPrefixes(256, 1)
    @test resolve_scan(AK.Auto(); dims=2) === AK.SliceScan(256, 8)

    # On the host backend: the threaded algorithm, filled
    @test resolve_scan(AK.Auto(); backend=HOST) === AK.CPUThreads.Partitioned(Threads.nthreads(), 2)
    @test resolve_scan(AK.Auto(); backend=HOST, dims=1) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 2)
end


@testset "scan resolution: explicit algorithms" begin
    # Explicit fields win over the tuning; `items_per_thread` is derived from the effective
    # `block_size`, so an explicit block size gets a matching default
    with_tuning(AK.ScanTuning(block_size=128, max_items=16)) do
        @test resolve_scan(AK.ScanPrefixes()) === AK.ScanPrefixes(128, 16)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024)) === AK.ScanPrefixes(1024, 7)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024, items_per_thread=12)) ===
              AK.ScanPrefixes(1024, 12)
        @test resolve_scan(AK.SliceScan(); dims=1) === AK.SliceScan(128, 16)
    end
    with_tuning(AK.ScanTuning(local_mem_bytes=1024)) do
        @test resolve_scan(AK.ScanPrefixes(block_size=64)) === AK.ScanPrefixes(64, 3)
        @test resolve_scan(AK.ScanPrefixes(block_size=1024)) === AK.ScanPrefixes(1024, 1)
    end
    @test resolve_scan(AK.CPUThreads.Partitioned(max_tasks=3); backend=HOST) ===
          AK.CPUThreads.Partitioned(3, 2)

    # Domains of explicit fields
    for bad in (AK.ScanPrefixes(block_size=0), AK.ScanPrefixes(block_size=96),
                AK.ScanPrefixes(block_size=2048), AK.ScanPrefixes(items_per_thread=0),
                AK.SliceScan(block_size=3), AK.SliceScan(items_per_thread=0))
        @test_throws ArgumentError resolve_scan(bad; dims=bad isa AK.SliceScan ? 1 : nothing)
    end
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(block_size=1024, items_per_thread=1 << 22))
    # A tuning's settings are checked like explicit ones
    for block_size in (0, 1 << 62)
        with_tuning(AK.ScanTuning(; block_size)) do
            @test_throws ArgumentError resolve_scan(AK.Auto())
        end
    end
    # Exclusive scans on threads need two elements per task
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.Partitioned(min_elems=1); backend=HOST)

    # Whole-array algorithms do not scan along `dims`, and SliceScan only does; they used to be
    # ignored silently along `dims`
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); dims=1)
    @test_throws ArgumentError resolve_scan(AK.SliceScan())

    # Tiles of partial results: lanes (a Bool and a value) are wider than elements, so fewer fit.
    # NTuple{3, Int32} is 4-byte aligned everywhere, so its lanes take 16 bytes on every platform
    # (Int64 lanes take 12 on 32-bit x86)
    @test resolve_scan(AK.Auto(); T=NTuple{3, Int32}) === AK.ScanPrefixes(256, 8)
    @test AK.resolve_scan(AK.Auto(), FAKE, NTuple{3, Int32}, nothing,
                          AK.Optional{NTuple{3, Int32}}) === AK.ScanPrefixes(256, 7)

    # Kernels need a bits-type element
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); T=String)
    @test_throws ArgumentError resolve_scan(AK.Auto(); T=Union{Missing, Int}, dims=1)

    # Capabilities: the threaded algorithm only on the host, kernels only where they run
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.Partitioned())
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); backend=NoKernelsBackend())

    # Algorithms of other families
    @test_throws ArgumentError resolve_scan(AK.BlockReduce())
    @test_throws ArgumentError resolve_scan(AK.CPUThreads.SampleSort(); backend=HOST)
    @test_throws ArgumentError AK.resolve_reduce(AK.ScanPrefixes(), FAKE, Float32, :)
end


@testset "scan resolution: inference" begin
    for (B, dims) in ((FakeBackend, Nothing), (FakeBackend, Int),
                      (typeof(HOST), Nothing), (typeof(HOST), Int))
        rt = only(Base.return_types(AK.resolve_scan, (AK.Auto, B, Type{Float32}, dims)))
        @test rt <: Union{AK.ScanPrefixes, AK.SliceScan, AK.CPUThreads.Partitioned}
    end
end


@testset "scan resolution: entry points" begin
    # The keyword backend: derived from the arrays, destination first
    v = rand(Int32(1):Int32(9), 1000)
    @test AK.accumulate!(+, copy(v); init=Int32(0)) == cumsum(v)
    @test AK.accumulate!(+, similar(v), v; init=Int32(0)) == cumsum(v)
    @test AK.accumulate(+, v; init=Int32(0), backend=HOST) == cumsum(v)
    # An explicit backend is used as given, and checked against the algorithm
    @test_throws ArgumentError AK.accumulate!(+, similar(v), v; init=Int32(0), backend=FAKE,
                                              alg=AK.CPUThreads.Partitioned())
    # A rejected algorithm leaves the destination untouched
    dst = zeros(Int32, 1000)
    @test_throws ArgumentError AK.accumulate!(+, dst, v; init=Int32(0),
                                              alg=AK.CPUThreads.Partitioned(min_elems=1))
    @test Base.all(iszero, dst)
    # Invalid `dims`
    @test_throws ArgumentError AK.accumulate(+, reshape(v, 10, 100); init=Int32(0), dims=0)
end


@testset "predicate resolution" begin
    @test AK.resolve_predicate(AK.Auto(), HOST, Float32) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    @test AK.resolve_predicate(AK.Auto(), FAKE, Float32) === AK.ConcurrentWrite(256)
    with_tuning(AK.PredicateTuning(prefer_concurrent_write=false)) do
        @test AK.resolve_predicate(AK.Auto(), FAKE, Float32) === AK.ViaReduce(AK.BlockReduce(256, 16, 0))
        # The nested reduction is resolved like any reduction
        @test AK.resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=64)), FAKE, Float32) ===
              AK.ViaReduce(AK.BlockReduce(64, 16, 0))
        @test_throws ArgumentError AK.resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=3)), FAKE, Float32)
        # A tuning only guides `Auto`
        @test AK.resolve_predicate(AK.ConcurrentWrite(), FAKE, Float32) === AK.ConcurrentWrite(256)
    end
    for bad in (AK.ConcurrentWrite(block_size=0), AK.ConcurrentWrite(block_size=96),
                AK.ConcurrentWrite(block_size=2048))
        @test_throws ArgumentError AK.resolve_predicate(bad, FAKE, Float32)
    end
    # Capabilities and other families
    @test_throws ArgumentError AK.resolve_predicate(AK.CPUThreads.Partitioned(), FAKE, Float32)
    @test_throws ArgumentError AK.resolve_predicate(AK.BlockReduce(), FAKE, Float32)
    @test_throws ArgumentError AK.resolve_predicate(AK.ScanScatter(), FAKE, Float32)
end


@testset "findall resolution" begin
    with_tuning(AK.FindallTuning(block_size=128, items_per_thread=4)) do
        @test AK.resolve_findall(AK.Auto(), FAKE, Bool) === AK.ScanScatter(128, 4)
        @test AK.resolve_findall(AK.ScanScatter(block_size=64), FAKE, Bool) === AK.ScanScatter(64, 4)
    end
    @test AK.resolve_findall(AK.Auto(), HOST, Bool) ===
          AK.CPUThreads.Partitioned(Threads.nthreads(), 1)
    for bad in (AK.ScanScatter(block_size=0), AK.ScanScatter(block_size=96),
                AK.ScanScatter(block_size=2048), AK.ScanScatter(items_per_thread=0),
                AK.ScanScatter(block_size=1024, items_per_thread=1 << 22))
        @test_throws ArgumentError AK.resolve_findall(bad, FAKE, Bool)
    end
    @test_throws ArgumentError AK.resolve_findall(AK.CPUThreads.Partitioned(), FAKE, Bool)
    @test_throws ArgumentError AK.resolve_findall(AK.BlockReduce(), FAKE, Bool)

    # Removed keywords
    v = rand(Bool, 10)
    @test_throws MethodError AK.findall(v; prefer_threads=true)
    @test_throws MethodError AK.findall(v; max_tasks=2)
    @test_throws MethodError AK.any(identity, v; prefer_threads=true)
    @test_throws MethodError AK.all(identity, v; block_size=64)
end
