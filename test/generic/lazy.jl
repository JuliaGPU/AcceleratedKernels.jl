using MappedArrays
using Adapt

# Tests that do not choose an algorithm use AK's kernels in the `--cpu-ka` configuration, as the
# other test files do
LAZY_REDUCE_ALG = HOST_KERNELS ? AK.BlockReduce() : AK.Auto()
LAZY_SORT_ALG = HOST_KERNELS ? AK.MergeSort() : AK.Auto()

# A lazy array computing its elements from one array, like MappedArrays' `mappedarray(f, data)`,
# but with an Adapt.jl rule, so that it can be passed to kernels when `data` is a device array
# (MappedArrays.jl does not define one). With `S`, `similar` gives an array like `data`, instead of
# Base's host `Array`.
struct LazyMap{T, N, F, A <: AbstractArray, S} <: AbstractArray{T, N}
    f::F
    data::A
end
LazyMap(f, data::AbstractArray{<:Any, N}; similar_data::Bool=false) where {N} =
    LazyMap{Base.promote_op(f, eltype(data)), N, typeof(f), typeof(data), similar_data}(f, data)
Base.size(a::LazyMap) = size(a.data)
Base.parent(a::LazyMap) = a.data
Base.IndexStyle(::Type{<:LazyMap{T, N, F, A}}) where {T, N, F, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(a::LazyMap, i::Int) = a.f(a.data[i])
Base.@propagate_inbounds Base.getindex(a::LazyMap{T, N}, I::Vararg{Int, N}) where {T, N} =
    a.f(a.data[I...])
Base.similar(a::LazyMap{<:Any, <:Any, <:Any, <:Any, true}, ::Type{T}, dims::Dims) where {T} =
    similar(a.data, T, dims)
function Adapt.adapt_structure(to, a::LazyMap{T, N, <:Any, <:Any, S}) where {T, N, S}
    f, data = adapt(to, a.f), adapt(to, a.data)
    LazyMap{T, N, typeof(f), typeof(data), S}(f, data)
end

# A host array wrapper with the parent `p`, which defines its own backend
struct TaggedTestBackend <: KernelAbstractions.Backend end
struct TaggedArray{T, N, P} <: AbstractArray{T, N}
    data::Array{T, N}
    p::P
end
TaggedArray(data) = TaggedArray(data, data)
Base.size(a::TaggedArray) = size(a.data)
Base.parent(a::TaggedArray) = a.p
Base.getindex(a::TaggedArray, i::Int...) = a.data[i...]
KernelAbstractions.get_backend(::TaggedArray) = TaggedTestBackend()

lazy_sq(x) = x * x
lazy_ci(I) = Int32(I[1] + 2 * I[2])
lazy_big(x) = x > 50
lazy_mod(x) = x % Int32(97)

# Run every read-only operation on the source `s`, comparing with Base on its host equivalent
# `h`, with the keywords `kw` (`backend`, or none); `B` is the backend the results must be on
function test_lazy_ops(s, h, B; kw...)
    T = eltype(h)
    on_backend(x) = get_backend(x) == B

    # Reductions
    @test AK.sum(s; alg=LAZY_REDUCE_ALG, kw...) == sum(h)
    @test AK.mapreduce(x -> 2x, +, s; alg=LAZY_REDUCE_ALG, init=T(10), kw...) ==
        mapreduce(x -> 2x, +, h; init=T(10))
    @test AK.maximum(s; alg=LAZY_REDUCE_ALG, kw...) == maximum(h)
    @test AK.count(lazy_big, s; alg=LAZY_REDUCE_ALG, kw...) == count(lazy_big, h)
    @test AK.any(lazy_big, s; kw...) == any(lazy_big, h)
    @test AK.all(x -> x > 0, s; kw...) == all(x -> x > 0, h)
    # ... with another array
    other = array_from_host(T.(eachindex(h)))
    @test AK.mapreduce(*, +, s, other; alg=LAZY_REDUCE_ALG, init=T(0), kw...) ==
        mapreduce(*, +, h, T.(eachindex(h)); init=T(0))

    # findall, of the source and of `items`
    r = AK.findall(lazy_big, s; kw...)
    @test on_backend(r) && Array(r) == findall(lazy_big, h)
    mask = array_from_host(isodd.(eachindex(h)))
    r = AK.findall(mask; items=s, kw...)
    @test on_backend(r) && Array(r) == h[isodd.(eachindex(h))]

    # Scans
    r = AK.cumsum(s; kw...)
    @test on_backend(r) && Array(r) == cumsum(h)
    r = AK.accumulate(+, s; init=T(0), kw...)
    @test on_backend(r) && Array(r) == accumulate(+, h; init=T(0))

    # map, reverse
    r = AK.map(x -> 2x, s; kw...)
    @test on_backend(r) && Array(r) == map(x -> 2x, h)
    dst = array_from_host(zeros(T, size(h)))
    @test Array(AK.map!(x -> 2x, dst, s; kw...)) == map(x -> 2x, h)
    r = AK.reverse(s; kw...)
    @test on_backend(r) && Array(r) == reverse(h)

    # Sorting copies the source
    r = AK.sort(s; alg=LAZY_SORT_ALG, kw...)
    @test on_backend(r) && Array(r) == sort(h)
    r = AK.sortperm(s; alg=LAZY_SORT_ALG, kw...)
    @test on_backend(r) && h[Array(r)] == sort(h)
    if TEST_KERNELS
        r = AK.sortperm(s; alg=AK.MergeSort(), kw...)
        @test h[Array(r)] == sort(h)
    end

    # Binary search, in the source and for its values
    sorted = sort(h)
    xs = array_from_host(T.(round.(Int, range(0, maximum(h) + 1; length=200))))
    ix = array_from_host(zeros(Int, length(xs)))
    AK.searchsortedfirst!(ix, issorted(h) ? s : array_from_host(sorted), xs; kw...)
    @test Array(ix) == searchsortedfirst.(Ref(sorted), Array(xs))
    ix = array_from_host(zeros(Int, length(h)))
    AK.searchsortedlast!(ix, array_from_host(sorted), s; kw...)
    @test Array(ix) == searchsortedlast.(Ref(sorted), h)
end

# Reductions and scans along dimensions of the matrix source `s`
function test_lazy_dims(s, h, B; kw...)
    for dims in (1, 2, (1, 2))
        r = AK.mapreduce(x -> 2x, +, s; dims, init=Int32(10), alg=LAZY_REDUCE_ALG, kw...)
        @test get_backend(r) == B && Array(r) == mapreduce(x -> 2x, +, h; dims, init=Int32(10))
    end
    R = array_from_host(ones(Int32, 1, size(h, 2)))
    AK.mapreducedim!(identity, +, R, s; alg=LAZY_REDUCE_ALG, kw...)
    @test Array(R) == 1 .+ sum(h; dims=1)
    for dims in (1, 2)
        r = AK.accumulate(+, s; dims, init=Int32(0), kw...)
        @test get_backend(r) == B && Array(r) == accumulate(+, h; dims, init=Int32(0))
    end
    for dims in (:, 1, 2)
        r = AK.reverse(s; dims, kw...)
        @test get_backend(r) == B && Array(r) == reverse(h; dims)
    end
end


@testset "lazy sources: backend resolution" begin
    host = AK.HOST_BACKEND
    d = array_from_host(Int32.(1:6))
    # A lazy array over ranges or Cartesian indices does not determine the backend...
    @test AK._resolve_backend(nothing, mappedarray(lazy_sq, 1:6)) == host
    @test AK._resolve_backend(nothing, mappedarray(lazy_ci, CartesianIndices((2, 3)))) == host
    @test AK._resolve_backend(nothing, mappedarray(+, 1:6, 2:7)) == host
    @test AK._resolve_backend(nothing, mappedarray(lazy_sq, reshape(1:6, 2, 3))) == host
    @test AK._resolve_backend(nothing, LazyMap(lazy_sq, mappedarray(lazy_sq, 1:6))) == host
    @test AK._resolve_backend(nothing, d, mappedarray(lazy_sq, 1:6)) == BACKEND
    # ... while one over arrays votes with them, whatever it wraps them in
    @test AK._resolve_backend(nothing, mappedarray(lazy_sq, d)) == BACKEND
    @test AK._resolve_backend(nothing, mappedarray(+, d, 1:6)) == BACKEND
    @test AK._resolve_backend(nothing, mappedarray(+, 1:6, d)) == BACKEND
    @test AK._resolve_backend(nothing, LazyMap(lazy_sq, mappedarray(+, 1:6, d))) == BACKEND
    if BACKEND != host
        @test_throws ArgumentError AK._resolve_backend(nothing, mappedarray(+, d, Int32.(1:6)))
    end
    @test AK._resolve_backend(BACKEND, mappedarray(lazy_sq, 1:6)) == BACKEND
    # A wrapper's own `get_backend` method is used
    @test AK._resolve_backend(nothing, TaggedArray(zeros(2))) === TaggedTestBackend()
    @test AK._resolve_backend(nothing, TaggedArray(zeros(2), 1:2)) === TaggedTestBackend()
    @test AK._resolve_backend(nothing, TaggedArray(zeros(2), (zeros(2), 1:2))) ===
        TaggedTestBackend()
    @test AK._resolve_backend(nothing, mappedarray(lazy_sq, TaggedArray(zeros(2)))) ===
        TaggedTestBackend()

    # Results are allocated like the array holding the source's memory, or on the backend
    @test get_backend(AK._similar(BACKEND, mappedarray(lazy_sq, d))) == BACKEND
    @test get_backend(AK._similar(BACKEND, mappedarray(lazy_sq, 1:6))) == BACKEND
    @test AK._similar(host, mappedarray(+, 1:6, 1:6), Int8) isa Vector{Int8}
end


@testset "lazy sources over ranges" begin
    # As in #23
    @test AK.sum(1234:100_000; backend=BACKEND, alg=LAZY_REDUCE_ALG) == sum(1234:100_000)
    @test AK.sum(mappedarray(x -> x * x, 1234:100_000); backend=BACKEND, alg=LAZY_REDUCE_ALG) ==
        sum(x -> x * x, 1234:100_000)
    @test AK.mapreduce(x -> 2x, +, mappedarray(x -> x * x, 1234:100_000);
                       backend=BACKEND, init=Int64(10), alg=LAZY_REDUCE_ALG) ==
        sum(x -> 2 * x * x, 1234:100_000; init=Int64(10))

    # Without a backend, on the host
    @test AK.sum(mappedarray(lazy_sq, Int32(1):Int32(6))) == 91

    n = 1000
    for (s, h) in ((mappedarray(lazy_sq, Int32(1):Int32(n)), map(lazy_sq, Int32(1):Int32(n))),
                   (mappedarray(+, Int32(1):Int32(n), Int32(2):Int32(n + 1)), collect(Int32(3):Int32(2):Int32(2n + 1))),
                   (mappedarray(lazy_mod, Int32(1):Int32(n)), map(lazy_mod, Int32(1):Int32(n))))
        # The backend given explicitly...
        test_lazy_ops(s, h, BACKEND; backend=BACKEND)
        # ... and on the host without it, where the host backend has no kernels
        TEST_KERNELS || test_lazy_ops(s, h, BACKEND)
    end

    C = CartesianIndices((37, 81))
    test_lazy_dims(mappedarray(lazy_ci, C), map(lazy_ci, C), BACKEND; backend=BACKEND)
    R = reshape(Int32(1):Int32(37 * 81), 37, 81)
    test_lazy_dims(mappedarray(lazy_mod, R), map(lazy_mod, R), BACKEND; backend=BACKEND)
    TEST_KERNELS || test_lazy_dims(mappedarray(lazy_ci, C), map(lazy_ci, C), BACKEND)
    # CartesianIndices themselves, with offset axes (as in #23)
    C = CartesianIndices((12:345, 67:89))
    for dims in (:, 1, 2, (1, 2))
        r = AK.mapreduce(I -> I[1], +, C; dims, init=10, backend=BACKEND, alg=LAZY_REDUCE_ALG)
        @test (dims isa Colon ? r : Array(r)) == mapreduce(I -> I[1], +, C; dims, init=10)
    end
end


@testset "lazy sources over arrays" begin
    n = 3000
    h = Int32.(rand(1:100, n))
    d = array_from_host(h)
    # A lazy array of the backend's arrays runs on their backend, given or not
    test_lazy_ops(LazyMap(lazy_sq, d), map(lazy_sq, h), BACKEND)
    test_lazy_ops(LazyMap(lazy_sq, d), map(lazy_sq, h), BACKEND; backend=BACKEND)
    # ... also when its `similar` gives the backend's arrays
    test_lazy_ops(LazyMap(lazy_sq, d; similar_data=true), map(lazy_sq, h), BACKEND)
    # ... and nested
    test_lazy_ops(LazyMap(x -> x + Int32(1), LazyMap(lazy_sq, d)), map(lazy_sq, h) .+ Int32(1), BACKEND)

    H = Int32.(rand(1:100, 37, 81))
    D = array_from_host(H)
    test_lazy_dims(LazyMap(lazy_sq, D), map(lazy_sq, H), BACKEND)
    test_lazy_dims(LazyMap(lazy_sq, view(D, 2:30, 3:70)), map(lazy_sq, H[2:30, 3:70]), BACKEND)

    if TEST_KERNELS
        # Reductions finishing on the host evaluate the lazy array on the device first
        alg = AK.BlockReduce(switch_below=100)
        @test AK.sum(LazyMap(lazy_sq, d[1:50]); alg) == sum(lazy_sq, h[1:50])
        @test AK.mapreduce(*, +, LazyMap(lazy_sq, d[1:50]), d[1:50]; alg) ==
            sum(map(lazy_sq, h[1:50]) .* h[1:50])
        @test AK.sum(LazyMap(lazy_sq, d[1:50]; similar_data=true); alg) == sum(lazy_sq, h[1:50])
    else
        # MappedArrays' own types can only be passed to kernels over ranges and other isbits
        # arrays (MappedArrays.jl has no Adapt.jl rules), but work with the host's arrays
        test_lazy_ops(mappedarray(lazy_sq, d), map(lazy_sq, h), BACKEND)
        test_lazy_dims(mappedarray(lazy_sq, D), map(lazy_sq, H), BACKEND)
    end
end


@testset "lazy sources: copies of host wrappers" begin
    # Base's copies, which do not race on packed storage, are used on the host backend (the
    # sort itself runs on one task: sorting a `BitVector` on several races)
    alg = AK.CPUThreads.SampleSort(max_tasks=1)
    for _ in 1:20
        bits = view(BitVector(rand(Bool, 1000)), :)
        @test AK.sort(bits; backend=AK.HOST_BACKEND, alg) == sort(collect(bits))
        @test bits[AK.sortperm(bits; backend=AK.HOST_BACKEND)] == sort(collect(bits))
        # ... and results are not allocated in packed storage, which `map!` would write to from
        # several tasks
        lazybits = LazyMap(identity, BitVector(rand(Bool, 1000)))
        r = AK.map(identity, lazybits; backend=AK.HOST_BACKEND)
        @test r isa Vector{Bool} && r == lazybits
    end
end
