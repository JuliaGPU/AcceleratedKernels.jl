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

# An array of which resolution only sees the size
struct SizeOnly{T, N} <: AbstractArray{T, N}
    dims::NTuple{N, Int}
end
Base.size(a::SizeOnly) = a.dims

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

    # A tuning without thresholds: merge sort, filled as an explicit one; on the host backend,
    # the threaded sample sort
    for kw in ((;), (perm=true,), (pairs=true,))
        @test resolve(AK.Auto(), v; kw...) === resolve(AK.MergeSort(), v; kw...)
        @test resolve(AK.Auto(), v; backend=HOST, kw...) ===
              resolve(AK.CPUThreads.SampleSort(), v; backend=HOST, kw...)
    end
    @test resolve(AK.Auto(), zeros(Int32, 10^6)) isa AK.MergeSort
    @test resolve(AK.Auto(), zeros(Int32, 10, 10); dims=1) isa AK.MergeSort

    # Thresholds: bitonic sort up to and including `bitonic_max_len`, radix sort from
    # `radix_min_len`, for element types and orderings where they give the same result; along
    # `dims`, the slice thresholds apply to the slices' length
    with_tuning(AK.SortTuning(bitonic_max_len=1024, radix_min_len=4096, bitonic_max_slice_len=64,
                              radix_min_slice_len=256, radix_block_size=128,
                              radix_items_per_thread=4, radix_chunk_size=64, bitonic_block_size=64,
                              bitonic_items_per_thread=4)) do
        @test resolve(AK.Auto(), zeros(Int32, 1024)) === AK.BitonicSort(64, 4)
        @test resolve(AK.Auto(), zeros(Int32, 1025)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4095)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4096)) === AK.RadixSort(128, 4, 64)
        @test resolve(AK.Auto(), zeros(Int32, 4096); ord=REV) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 64, 64)) isa AK.RadixSort      # dims=: sorts flat

        # Along `dims`, contiguous or strided slices
        for (sz, d) in (((64, 16), 1), ((16, 64), 2), ((2, 64, 8), 2))
            @test resolve(AK.Auto(), zeros(Int32, sz); dims=d) === AK.BitonicSort(64, 4)
        end
        @test resolve(AK.Auto(), zeros(Int32, 65, 16); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 256, 16); dims=1) === AK.RadixSort(128, 4, 64)
        @test resolve(AK.Auto(), zeros(Int32, 16, 256); dims=2) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 256, 16); dims=1, ord=REV) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 255, 32); dims=1) isa AK.MergeSort
        # ... where the array also has at least `radix_min_len` elements
        @test resolve(AK.Auto(), zeros(Int32, 256, 15); dims=1) isa AK.MergeSort

        # A single slice sorts like a whole array
        @test resolve(AK.Auto(), zeros(Int32, 1024); dims=1) isa AK.BitonicSort
        @test resolve(AK.Auto(), zeros(Int32, 1024, 1); dims=1) isa AK.BitonicSort
        @test resolve(AK.Auto(), zeros(Int32, 2048); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000); dims=1) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000, 1); dims=1) isa AK.RadixSort
        @test resolve(AK.Auto(), zeros(Int32, 1, 10_000); dims=2) isa AK.RadixSort

        # ... of at most 2^32 elements per array or slice
        if Sys.WORD_SIZE == 64
            @test resolve(AK.Auto(), SizeOnly{Int32, 1}((2^32 + 1,))) isa AK.MergeSort
            @test resolve(AK.Auto(), SizeOnly{Int32, 2}((2^32 + 1, 2)); dims=1) isa AK.MergeSort
            @test resolve(AK.Auto(), SizeOnly{Int32, 1}((2^32,))) isa AK.RadixSort
            @test_throws ArgumentError resolve(AK.RadixSort(), SizeOnly{Int32, 1}((2^32 + 1,)))
        end

        # Radix sort needs a supported element type and the default ordering or its reverse
        @test resolve(AK.Auto(), zeros(Int16, 10_000)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000); ord=Base.Order.ord(isless, abs, nothing)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 10_000); ord=Base.Order.ord(>, identity, nothing)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int16, 256, 16); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 256, 16); dims=1, ord=Base.Order.ord(isless, abs, nothing)) isa AK.MergeSort

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
        @test resolve(AK.Auto(), zeros(Float32, 64, 16); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Float32, 64, 16); dims=1) isa AK.BitonicSort

        # Bitonic sort only where its tiles (here of 256 elements) fit in local memory
        @test resolve(AK.Auto(stable=false), Vector{NTuple{16, Float64}}(undef, 100)) isa AK.BitonicSort
        @test resolve(AK.Auto(stable=false), Vector{NTuple{17, Float64}}(undef, 100)) isa AK.MergeSort

        # sortperm! and key/value sorting always observe stability, and need a permutation path
        @test resolve(AK.Auto(), zeros(Int32, 100); perm=true) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Float32, 100); perm=true) isa AK.MergeSort
        @test resolve(AK.Auto(stable=false), zeros(Int32, 10_000); pairs=true) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 64, 16); dims=1, perm=true) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 256, 16); dims=1, pairs=true) isa AK.MergeSort
    end

    # The whole-array thresholds do not apply along `dims`, nor the slice thresholds to whole
    # arrays
    with_tuning(AK.SortTuning(bitonic_max_len=1024, radix_min_len=0)) do
        @test resolve(AK.Auto(), zeros(Int32, 64, 16); dims=1) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4096, 16); dims=1) isa AK.MergeSort
    end
    with_tuning(AK.SortTuning(bitonic_max_slice_len=1024, radix_min_slice_len=0)) do
        @test resolve(AK.Auto(), zeros(Int32, 64)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4096)) isa AK.MergeSort
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

        # Merge sort's default block size shrinks until its tiles fit in local memory
        big = Vector{NTuple{32, Float64}}(undef, 16)                # 256 bytes
        @test resolve(AK.MergeSort(), big) === AK.MergeSort(64, false)
        @test resolve(AK.Auto(), big) === AK.MergeSort(64, false)
        @test resolve(AK.MergeSort(), big; perm=true, merge_eltypes=(eltype(big), Int)) ===
              AK.MergeSort(32, false)
        @test resolve(AK.MergeSort(lowmem=true), big; perm=true,
                      merge_eltypes=(eltype(big), Int)) === AK.MergeSort(128, true)
    end
    # On the host, `max_tasks` defaults to the number of threads
    min_elems = AK.sort_tuning(HOST, Float32).threads_min_elems
    @test resolve(AK.CPUThreads.SampleSort(max_tasks=3), v; backend=HOST) ===
          AK.CPUThreads.SampleSort(3, min_elems)
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
    @test rejects(AK.MergeSort(block_size=100))
    with_tuning(AK.SortTuning(merge_block_size=96)) do             # tuned sizes are checked too
        @test rejects(AK.MergeSort())
        @test rejects(AK.Auto(); perm=true)
    end
    @test rejects(AK.MergeSort(lowmem=true))                        # sort!
    @test rejects(AK.MergeSort(lowmem=true); pairs=true)            # sort_by_key!
    @test !rejects(AK.MergeSort(lowmem=true); perm=true)
    @test !rejects(AK.MergeSort(), zeros(Float32, 4, 4); dims=2, perm=true)
    @test !rejects(AK.MergeSort(block_size=4096))                   # tiles of up to 32 KiB
    @test rejects(AK.MergeSort(block_size=8192))
    @test rejects(AK.MergeSort(block_size=2048); perm=true, merge_eltypes=(Float32, Int64))
    @test !rejects(AK.MergeSort(block_size=2048, lowmem=true); perm=true,
                   merge_eltypes=(Float32, Int64))                  # only the indices
    @test rejects(AK.MergeSort(block_size=1), Vector{NTuple{2049, Float64}}(undef, 16))

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
    @test !rejects(AK.BitonicSort(block_size=1024, items_per_thread=8), zeros(Float32, 16))
    @test rejects(AK.BitonicSort(block_size=1024, items_per_thread=8), zeros(Int64, 16))  # local memory
    @test rejects(AK.BitonicSort(block_size=1024, items_per_thread=16), zeros(Float32, 16))
    @test rejects(AK.BitonicSort(), Vector{NTuple{4, Float64}}(undef, 16))
    @test !rejects(AK.BitonicSort(items_per_thread=4), Vector{NTuple{4, Float64}}(undef, 16))
    @test rejects(AK.BitonicSort(block_size=1024, items_per_thread=4), Vector{Union{Missing, Int64}}(undef, 16))

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
    # Neither do Base's views, reshapes and permutations of those leaves; those of arrays do
    @test AK.resolve_backend(nothing, d, reshape(1:6, 2, 3), view(1:10, 2:4),
                             PermutedDimsArray(reshape(1:6, 2, 3), (2, 1))) === FAKE
    @test AK.resolve_backend(nothing, view(d, 1:2)) === FAKE
    @test AK.resolve_backend(nothing, reshape(1:6, 2, 3)) == HOST

    # All votes must agree, including the non-destination inputs
    @test_throws ArgumentError AK.resolve_backend(nothing, d, h)
    @test_throws ArgumentError AK.resolve_backend(nothing, h, d)
    @test_throws ArgumentError AK.resolve_backend(nothing, h, (1, d))
    # An explicit backend wins over the votes
    @test AK.resolve_backend(FAKE, h) === FAKE
    @test AK.resolve_backend(HOST, d, h) == HOST

    # Broadcasted trees are walked
    bc = Base.Broadcast.broadcasted(+, d, Base.Broadcast.broadcasted(*, 2, 1:4))
    @test AK.resolve_backend(nothing, bc) === FAKE
    @test_throws ArgumentError AK.resolve_backend(nothing, h, bc)
    # Also the arrays of a preprocessed one
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
    # Allocating operations given the backend do not ask the arrays either
    @test AK.sortperm(NoBackendArray(); backend=HOST) == [1, 2, 3]
    @test AK.backendless(reshape(1:6, 2, 3)) && !AK.backendless(NoBackendArray())

    # The sorting entry points resolve from all their arrays, and accept only backends
    @test_throws ArgumentError AK.sortperm!(zeros(Int, 4), d)
    @test_throws ArgumentError AK.sort_by_key!(d, h)
    @test_throws TypeError AK.sort!(h; backend=:cpu)

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
    # On a GPU, BlockReduce, whose every field the tuning fills, for whole arrays and along `dims`;
    # `stable` means nothing to a reduction
    with_tuning(AK.ReduceTuning(block_size=512, items_per_thread=4, switch_below=100)) do
        for dims in (:, nothing, 2, (1, 3))
            @test resolve_reduce(AK.Auto(); dims) === AK.BlockReduce(512, 4, 100)
        end
        @test resolve_reduce(AK.Auto(stable=false)) === AK.BlockReduce(512, 4, 100)
    end

    # On the host backend: the threaded algorithm, filled
    for dims in (:, 1)
        @test resolve_reduce(AK.Auto(); backend=HOST, dims) ===
              resolve_reduce(AK.CPUThreads.Partitioned(); backend=HOST, dims)
    end

    # A tuning without `items_per_thread` loads 64 bytes of accumulators per thread and tile, 4 to
    # 16 elements
    with_tuning(AK.ReduceTuning(block_size=512, switch_below=100)) do
        for (T, items_per_thread) in ((Int8, 16), (Float32, 16), (Float64, 8),
                                      (NTuple{8, Float64}, 4), (Union{}, 4))
            @test resolve_reduce(AK.Auto(); T) === AK.BlockReduce(512, items_per_thread, 100)
        end
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
          AK.CPUThreads.Partitioned(3, AK.reduce_tuning(HOST, Float32).threads_min_elems)
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

    # `items_per_thread` and `switch_below` only apply to whole-array reductions
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=1)
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(switch_below=0); dims=(1, 2))
    @test resolve_reduce(AK.BlockReduce(items_per_thread=4); dims=nothing) isa AK.BlockReduce

    # Whole-array reductions need tiles of at least two elements (one never shrinks the input)
    # and at most typemax(Int32), whether the settings are explicit or from the tuning
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=1))
    @test resolve_reduce(AK.BlockReduce(block_size=1, items_per_thread=2)).items_per_thread == 2
    @test resolve_reduce(AK.BlockReduce(block_size=1); dims=1).block_size == 1
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(items_per_thread=1 << 56))
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(block_size=1024, items_per_thread=1 << 22))
    with_tuning(AK.ReduceTuning(block_size=1, items_per_thread=1, switch_below=0)) do
        @test_throws ArgumentError resolve_reduce(AK.Auto())
        @test resolve_reduce(AK.Auto(); dims=2) === AK.BlockReduce(1, 1, 0)
    end

    # Capabilities: the threaded algorithm only on the host, kernels only where they run
    @test_throws ArgumentError resolve_reduce(AK.CPUThreads.Partitioned())
    @test_throws ArgumentError resolve_reduce(AK.BlockReduce(); backend=NoKernelsBackend())
    if AK.runs_kernels(HOST)
        @test resolve_reduce(AK.BlockReduce(); backend=HOST) isa AK.BlockReduce
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
end


@testset "reduce resolution: accumulator types" begin
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
    # On a GPU, ScanPrefixes for whole arrays and SliceScan along `dims`, with the tuning's block
    # size and as many items per thread as fit its local memory, at most `max_items`
    with_tuning(AK.ScanTuning(block_size=256, local_mem_bytes=32 * 1024, max_items=8)) do
        @test resolve_scan(AK.Auto()) === AK.ScanPrefixes(256, 8)
        @test resolve_scan(AK.Auto(); T=ComplexF64) === AK.ScanPrefixes(256, 7)
        @test resolve_scan(AK.Auto(); T=NTuple{16, Float64}) === AK.ScanPrefixes(256, 1)
        @test resolve_scan(AK.Auto(); dims=2) === AK.SliceScan(256, 8)
    end

    # On the host backend: the threaded algorithm, filled
    for dims in (nothing, 1)
        @test resolve_scan(AK.Auto(); backend=HOST, dims) ===
              resolve_scan(AK.CPUThreads.Partitioned(); backend=HOST, dims)
    end
end


@testset "scan resolution: explicit algorithms" begin
    # Explicit fields win over the tuning; `items_per_thread` is derived from the effective
    # `block_size`, so an explicit block size gets a matching default
    with_tuning(AK.ScanTuning(block_size=128, local_mem_bytes=32 * 1024, max_items=16)) do
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
          AK.CPUThreads.Partitioned(3, AK.scan_tuning(HOST, Float32).threads_min_elems)

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

    # Whole-array algorithms do not scan along `dims`, and SliceScan only does
    @test_throws ArgumentError resolve_scan(AK.ScanPrefixes(); dims=1)
    @test_throws ArgumentError resolve_scan(AK.SliceScan())

    # Tiles of partial results: lanes (a Bool and a value) are wider than elements, so fewer fit.
    # NTuple{3, Int32} is 4-byte aligned everywhere, so its lanes take 16 bytes on every platform
    # (Int64 lanes take 12 on 32-bit x86)
    with_tuning(AK.ScanTuning(block_size=256, local_mem_bytes=32 * 1024, max_items=8)) do
        @test resolve_scan(AK.Auto(); T=NTuple{3, Int32}) === AK.ScanPrefixes(256, 8)
        @test AK.resolve_scan(AK.Auto(), FAKE, NTuple{3, Int32}, nothing,
                              AK.Optional{NTuple{3, Int32}}) === AK.ScanPrefixes(256, 7)
    end

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
          AK.resolve_predicate(AK.CPUThreads.Partitioned(), HOST, Float32)
    # On a GPU, ConcurrentWrite, or ViaReduce where the tuning prefers it, whose reduction is
    # resolved like any reduction
    with_tuning(AK.PredicateTuning(block_size=128)) do
        @test AK.resolve_predicate(AK.Auto(), FAKE, Float32) === AK.ConcurrentWrite(128)
    end
    with_tuning(AK.PredicateTuning(prefer_concurrent_write=false, block_size=128)) do
        @test AK.resolve_predicate(AK.Auto(), FAKE, Float32) ===
              AK.ViaReduce(AK.resolve_reduce(AK.BlockReduce(), FAKE, Bool, :))
        @test AK.resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=64)), FAKE, Float32) ===
              AK.ViaReduce(AK.resolve_reduce(AK.BlockReduce(block_size=64), FAKE, Bool, :))
        @test_throws ArgumentError AK.resolve_predicate(AK.ViaReduce(AK.BlockReduce(block_size=3)),
                                                        FAKE, Float32)
        # `prefer_concurrent_write` only guides `Auto`
        @test AK.resolve_predicate(AK.ConcurrentWrite(), FAKE, Float32) === AK.ConcurrentWrite(128)
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
          AK.resolve_findall(AK.CPUThreads.Partitioned(), HOST, Bool)
    for bad in (AK.ScanScatter(block_size=0), AK.ScanScatter(block_size=96),
                AK.ScanScatter(block_size=2048), AK.ScanScatter(items_per_thread=0),
                AK.ScanScatter(block_size=1024, items_per_thread=1 << 22))
        @test_throws ArgumentError AK.resolve_findall(bad, FAKE, Bool)
    end
    @test_throws ArgumentError AK.resolve_findall(AK.CPUThreads.Partitioned(), FAKE, Bool)
    @test_throws ArgumentError AK.resolve_findall(AK.BlockReduce(), FAKE, Bool)
end
