# Resolution of sorting algorithms and of the backend, without launching kernels: resolution only
# looks at types, sizes and orderings, so host arrays stand in for device arrays throughout.

# A GPU backend with a tuning the tests can change, and arrays that live on it
struct ResolveTestBackend <: KernelAbstractions.GPU end
const RESOLVE_TUNING = Ref(AK.SortTuning())
AK.sort_tuning(::ResolveTestBackend, ::Type) = RESOLVE_TUNING[]

struct ResolveTestArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(a::ResolveTestArray) = size(a.data)
Base.getindex(a::ResolveTestArray, i::Int...) = a.data[i...]
KernelAbstractions.get_backend(::ResolveTestArray) = ResolveTestBackend()

# A backend that cannot run AK's kernels, like KernelAbstractions 0.9's `CPU`
struct NoKernelsTestBackend <: KernelAbstractions.GPU end
AK._runs_kernels(::NoKernelsTestBackend) = false

# An algorithm that is not a sorting algorithm
struct NotASortTestAlgorithm <: AK.Algorithm end

# An array type that does not implement `get_backend`
struct NoBackendTestArray <: AbstractVector{Int} end
Base.size(::NoBackendTestArray) = (3,)
Base.getindex(::NoBackendTestArray, i::Int) = i

# A host vector that must not be asked for its backend
struct NoVoteTestVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(a::NoVoteTestVector) = size(a.data)
Base.getindex(a::NoVoteTestVector, i::Int) = a.data[i]
Base.setindex!(a::NoVoteTestVector, x, i::Int) = (a.data[i] = x)
KernelAbstractions.get_backend(::NoVoteTestVector) = error("backend queried despite an explicit backend")

const RB = ResolveTestBackend()
const FWD = Base.Order.Forward
const REV = Base.Order.Reverse

resolve(alg, v; dims=:, ord=FWD, backend=RB, kw...) = AK._resolve_sort(alg, backend, v, dims, ord; kw...)

function with_tuning(f; kwargs...)
    old = RESOLVE_TUNING[]
    RESOLVE_TUNING[] = AK.SortTuning(; kwargs...)
    try
        f()
    finally
        RESOLVE_TUNING[] = old
    end
end


@testset "sort resolution: Auto" begin
    v = zeros(Float32, 1000)

    # The default tuning reproduces AK's historical choices and settings
    @test resolve(AK.Auto(), v) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), v; perm=true) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), v; pairs=true) === AK.MergeSort(256, false)
    @test resolve(AK.Auto(), zeros(Int32, 10^8 ÷ 100)) isa AK.MergeSort
    @test resolve(AK.Auto(), zeros(Int32, 10, 10); dims=1) isa AK.MergeSort

    # On the host backend: the threaded sample sort, filled
    host = AK.HOST_BACKEND
    a = resolve(AK.Auto(), v; backend=host)
    @test a isa AK.CPUThreads.SampleSort
    @test a.max_tasks == Threads.nthreads() && a.min_elems == 1
    @test resolve(AK.Auto(), v; backend=host, perm=true) isa AK.CPUThreads.SampleSort
    @test resolve(AK.Auto(), v; backend=host, pairs=true) isa AK.CPUThreads.SampleSort

    # Thresholds: bitonic sort up to and including `bitonic_max_len`, radix sort from
    # `radix_min_len`, for element types and orderings where they give the same result
    with_tuning(; bitonic_max_len=1024, radix_min_len=4096, radix_block_size=128,
                  radix_items_per_thread=4, bitonic_block_size=64, bitonic_items_per_thread=4) do
        @test resolve(AK.Auto(), zeros(Int32, 1024)) === AK.BitonicSort(64, 4)
        @test resolve(AK.Auto(), zeros(Int32, 1025)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4095)) isa AK.MergeSort
        @test resolve(AK.Auto(), zeros(Int32, 4096)) === AK.RadixSort(128, 4)
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
    with_tuning(; radix_min_len=0) do
        @test resolve(AK.Auto(), zeros(Float32, 10)) isa AK.RadixSort
    end
    @test resolve(AK.Auto(), zeros(Float32, 10)) isa AK.MergeSort

    # Selection never depends on the data
    @test resolve(AK.Auto(), fill(NaN32, 100)) === resolve(AK.Auto(), zeros(Float32, 100))
end


@testset "sort resolution: explicit algorithms" begin
    v = zeros(Float32, 1000)

    # Explicit fields win over the tuning, unset ones come from it
    with_tuning(; merge_block_size=128, radix_block_size=512, radix_items_per_thread=8,
                  bitonic_block_size=32, bitonic_items_per_thread=16, threads_min_elems=7) do
        @test resolve(AK.MergeSort(), v) === AK.MergeSort(128, false)
        @test resolve(AK.MergeSort(block_size=64), v) === AK.MergeSort(64, false)
        @test resolve(AK.MergeSort(lowmem=true), v; perm=true) === AK.MergeSort(128, true)
        @test resolve(AK.RadixSort(), v) === AK.RadixSort(512, 8)
        @test resolve(AK.RadixSort(items_per_thread=1), v) === AK.RadixSort(512, 1)
        @test resolve(AK.BitonicSort(block_size=128), v) === AK.BitonicSort(128, 16)
    end
    host = AK.HOST_BACKEND
    @test resolve(AK.CPUThreads.SampleSort(max_tasks=3), v; backend=host) ===
          AK.CPUThreads.SampleSort(3, 1)
    @test resolve(AK.CPUThreads.SampleSort(min_elems=5), v; backend=host) ===
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
    @test rejects(AK.RadixSort(), zeros(Float32, 4, 4); dims=1)
    @test rejects(AK.RadixSort(), zeros(Float32, 16); dims=1)
    @test rejects(AK.RadixSort(), zeros(Int16, 16))
    @test rejects(AK.RadixSort(), Vector{Tuple{Int32, Int32}}(undef, 16))
    @test rejects(AK.RadixSort(); ord=Base.Order.ord(isless, abs, nothing))
    @test rejects(AK.RadixSort(); ord=Base.Order.ord(>, identity, nothing))
    @test rejects(AK.RadixSort(block_size=8192))                    # local memory
    @test rejects(AK.RadixSort(block_size=1 << 62))                  # no overflowing footprints
    @test rejects(AK.RadixSort(items_per_thread=1 << 62))
    @test rejects(AK.RadixSort(items_per_thread=65))
    for T in (UInt32, Int32, Float32, UInt64, Int64, Float64), ord in (FWD, REV)
        @test !rejects(AK.RadixSort(), zeros(T, 16); ord)
    end
    @test !rejects(AK.RadixSort(), zeros(Float32, 4, 4))             # dims=: sorts flat

    # BitonicSort
    @test rejects(AK.BitonicSort(block_size=100))
    @test rejects(AK.BitonicSort(items_per_thread=3))
    @test rejects(AK.BitonicSort(block_size=2, items_per_thread=1 << (Sys.WORD_SIZE - 2)))
    @test rejects(AK.BitonicSort(); perm=true)
    @test rejects(AK.BitonicSort(); pairs=true)
    @test !rejects(AK.BitonicSort(), zeros(Float32, 4, 4); dims=2, ord=Base.Order.ord(isless, abs, nothing))

    # CPUThreads.SampleSort only on the host, where every operation supports it
    @test rejects(AK.CPUThreads.SampleSort())
    @test rejects(AK.CPUThreads.SampleSort(max_tasks=0); backend=host)
    @test rejects(AK.CPUThreads.SampleSort(min_elems=0); backend=host)
    @test !rejects(AK.CPUThreads.SampleSort(); backend=host, perm=true)
    @test !rejects(AK.CPUThreads.SampleSort(); backend=host, pairs=true, dims=1)

    # Kernel algorithms need a backend that runs AK's kernels
    for alg in (AK.MergeSort(), AK.RadixSort(), AK.BitonicSort())
        @test rejects(alg; backend=NoKernelsTestBackend())
        @test rejects(alg; backend=host) == !AK._runs_kernels(host)
    end

    # Domain checks run before anything else, whatever the input length
    @test rejects(AK.MergeSort(block_size=-1), zeros(Float32, 0))
    @test rejects(AK.RadixSort(block_size=3), zeros(Float32, 1))

    # Invalid `dims`
    @test rejects(AK.Auto(), zeros(Float32, 4, 4); dims=3)
    @test rejects(AK.Auto(), zeros(Float32, 4, 4); dims=0)

    # Not a sorting algorithm
    @test rejects(NotASortTestAlgorithm())
end


@testset "sort resolution: inference" begin
    for T in (Int32, Float32), (A, D) in ((Vector{T}, Colon), (Matrix{T}, Int))
        # Auto resolves to a small union of concrete algorithms, not `Any`
        rt = only(Base.return_types(AK._resolve_sort,
                                    (AK.Auto, ResolveTestBackend, A, D, Base.Order.ForwardOrdering)))
        @test rt <: Union{AK.MergeSort, AK.RadixSort, AK.BitonicSort}
        @test rt isa Union || isconcretetype(rt)
        rt = only(Base.return_types(AK._resolve_sort,
                                    (AK.Auto, typeof(AK.HOST_BACKEND), A, D, Base.Order.ForwardOrdering)))
        @test rt === AK.CPUThreads.SampleSort
        # An explicit algorithm resolves to itself, or always throws (no radix sort along `dims`)
        rt = only(Base.return_types(AK._resolve_sort,
                                    (AK.RadixSort, ResolveTestBackend, A, D, Base.Order.ForwardOrdering)))
        @test rt === (D === Colon ? AK.RadixSort : Union{})
    end
end


@testset "sort resolution: tunings of $(nameof(typeof(BACKEND)))" begin
    # Every tuning this backend's device returns resolves to algorithms that pass the checks
    for T in valid_backend_eltypes(BACKEND, (UInt8, Int16, Int32, UInt32, Float32, Int64, Float64))
        for len in (1, 100, 10_000, 10^7), stable in (true, false)
            v = Vector{T}(undef, len)       # only the type and length matter
            @test resolve(AK.Auto(; stable), v; backend=BACKEND) isa AK.SortAlgorithm
            @test resolve(AK.Auto(; stable), v; backend=BACKEND, perm=true) isa AK.SortAlgorithm
        end
        if AK._runs_kernels(BACKEND)
            for alg in (AK.MergeSort(), AK.BitonicSort(), AK.RadixSort())
                alg isa AK.RadixSort && !AK._rs_supported(T) && continue
                @test resolve(alg, zeros(T, 16); backend=BACKEND) isa typeof(alg)
            end
        end
    end
end


@testset "backend resolution" begin
    d = ResolveTestArray(zeros(Float32, 4))
    h = zeros(Float32, 4)
    host = AK.HOST_BACKEND

    # Every array votes, destination first; backend-free leaves do not
    @test AK._resolve_backend(nothing, d) === RB
    @test AK._resolve_backend(nothing, d, 1:4, CartesianIndices((2, 2)), LinearIndices(h),
                              1.0, Ref(2), (3, 1:2), nothing) === RB
    @test AK._resolve_backend(nothing, 1:4, d) === RB
    @test AK._resolve_backend(nothing, h, 1:4) == host

    # All votes must agree, including the non-destination inputs
    @test_throws ArgumentError AK._resolve_backend(nothing, d, h)
    @test_throws ArgumentError AK._resolve_backend(nothing, h, d)
    @test_throws ArgumentError AK._resolve_backend(nothing, h, (1, d))
    # ... unless the backend is explicit
    @test AK._resolve_backend(RB, h) === RB
    @test AK._resolve_backend(host, d, h) == host

    # Broadcasted trees are walked
    bc = Base.Broadcast.broadcasted(+, d, Base.Broadcast.broadcasted(*, 2, 1:4))
    @test AK._resolve_backend(nothing, bc) === RB
    @test_throws ArgumentError AK._resolve_backend(nothing, h, bc)
    # ... including the arrays of a preprocessed one
    pbc = Base.Broadcast.preprocess(nothing,
        Base.Broadcast.instantiate(Base.Broadcast.broadcasted(identity, d)))
    @test AK._resolve_backend(nothing, pbc) === RB
    @test_throws ArgumentError AK._resolve_backend(nothing, h, pbc)

    # Nothing votes: the host backend
    @test AK._resolve_backend(nothing) == host
    @test AK._resolve_backend(nothing, 1:10, 2.0) == host

    # Array types without `get_backend` keep raising KernelAbstractions' error
    @test_throws ArgumentError AK._resolve_backend(nothing, NoBackendTestArray())
    @test AK._resolve_backend(RB, NoBackendTestArray()) === RB
    # ... and allocating operations, given the backend, do not ask for it either
    @test AK.sortperm(NoBackendTestArray(); backend=host) == [1, 2, 3]
    @test AK._backend_free(reshape(1:6, 2, 3)) && !AK._backend_free(NoBackendTestArray())

    # The sorting entry points resolve from all their arrays, and accept only backends
    @test_throws ArgumentError AK.sort!(d; temp=h)
    @test_throws ArgumentError AK.sortperm!(zeros(Int, 4), d)
    @test_throws ArgumentError AK.sort_by_key!(d, h)
    @test_throws TypeError AK.sort!(h; backend=:cpu)
    # ... and never take a positional backend
    @test_throws MethodError AK.sort!(h, host)

    # Nested operations receive the resolved backend instead of querying the arrays again
    alg = AK.CPUThreads.SampleSort(max_tasks=4)
    v = NoVoteTestVector(Int32[3, 1, 2])
    @test AK.sort!(v; backend=host, alg).data == [1, 2, 3]
    ix = NoVoteTestVector(zeros(Int, 100))
    vals = rand(Int32, 100)
    @test AK.sortperm!(ix, vals; backend=host, alg).data == sortperm(vals)
    k = NoVoteTestVector(rand(Int32(1):Int32(3), 100))
    kv = copy(k.data)
    AK.sort_by_key!(k, NoVoteTestVector(collect(1:100)); backend=host, alg)
    @test k.data == sort(kv)
end
