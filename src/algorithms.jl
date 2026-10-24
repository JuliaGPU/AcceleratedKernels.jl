# Algorithms, capabilities and backend resolution: the layer every operation's host API shares.


"""
    Algorithm

Supertype of the algorithms accepted by AcceleratedKernels' operations through their `alg`
keyword. [`Auto`](@ref) lets AK choose; a concrete algorithm (e.g. [`MergeSort`](@ref)) is
honoured or rejected with an `ArgumentError`, never silently replaced.

Algorithms carry their tunable settings as fields; a field left at `nothing` is filled in from
AK's defaults for the device the operation runs on.
"""
abstract type Algorithm end


"""
    Auto(; stable=true)

Let AcceleratedKernels choose the algorithm and its settings for the current device, subject to
the requirements given as fields. This is the default `alg` of every operation.

`stable` applies to sorting only (other operations ignore it): with `stable=true` the result is
the one a stable sort would produce. An unstable algorithm is chosen only where no one can tell
the difference, i.e. where elements that compare equal are bitwise identical (integers, `Bool` and
`Char` under the default ordering). `stable=false` also allows unstable algorithms for other
element types; this is what Base's `alg=QuickSort` means.

Selection never reads the array's contents.
"""
Base.@kwdef struct Auto <: Algorithm
    stable::Bool = true
end


# Algorithm families, for documentation and dispatch. An operation's `alg` keyword accepts any
# `Algorithm`, and rejects those that do not implement it.

"""
    SortAlgorithm <: Algorithm

Supertype of the sorting algorithms: [`MergeSort`](@ref), [`RadixSort`](@ref),
[`BitonicSort`](@ref) and [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort).
"""
abstract type SortAlgorithm <: Algorithm end

"""
    ReduceAlgorithm <: Algorithm

Supertype of the reduction algorithms: [`BlockReduce`](@ref). Reductions also accept
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned).
"""
abstract type ReduceAlgorithm <: Algorithm end

"""
    ScanAlgorithm <: Algorithm

Supertype of the scan (`accumulate`) algorithms: [`ScanPrefixes`](@ref) and
[`SliceScan`](@ref). Scans also accept
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned).
"""
abstract type ScanAlgorithm <: Algorithm end

"""
    FindallAlgorithm <: Algorithm

Supertype of the stream-compaction (`findall`) algorithms: [`ScanScatter`](@ref). `findall`
also accepts [`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned).
"""
abstract type FindallAlgorithm <: Algorithm end

"""
    PredicateAlgorithm <: Algorithm

Supertype of the algorithms of `any` and `all`: [`ConcurrentWrite`](@ref) and
[`ViaReduce`](@ref). They also accept
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned).
"""
abstract type PredicateAlgorithm <: Algorithm end


"""
    AcceleratedKernels.CPUThreads

Algorithms that run on Julia threads, for arrays on the host backend.
[`Auto`](@ref AcceleratedKernels.Auto) chooses them for host arrays; they are rejected for any
other backend.
"""
module CPUThreads

import ..AcceleratedKernels: Algorithm, SortAlgorithm

"""
    CPUThreads.SampleSort(; max_tasks=nothing, min_elems=nothing)

Parallel sample sort on Julia threads, deferring to `Base.sort!` for the local sorts; it is
stable, and also provides `sortperm!` and [`sort_by_key!`](@ref AcceleratedKernels.sort_by_key!).
Uses at most `max_tasks` tasks (default `Threads.nthreads()`), each with at least `min_elems`
elements (default 1). Only runs on the host backend.
"""
Base.@kwdef struct SampleSort <: SortAlgorithm
    max_tasks::Union{Nothing, Int} = nothing
    min_elems::Union{Nothing, Int} = nothing
end

"""
    CPUThreads.Partitioned(; max_tasks=nothing, min_elems=nothing)

Split the input into contiguous parts, one per task, process them on Julia threads, and combine
the parts' results. Used by reductions, scans, `findall` and `any`/`all` on host arrays. Uses at
most `max_tasks` tasks (default `Threads.nthreads()`), each with at least `min_elems` elements
(default 1, or the operation's tuning). Only runs on the host backend.
"""
Base.@kwdef struct Partitioned <: Algorithm
    max_tasks::Union{Nothing, Int} = nothing
    min_elems::Union{Nothing, Int} = nothing
end

end # module CPUThreads


# Capabilities: correctness facts about a backend, checked for `Auto` and explicit algorithms
# alike. Unlike tunings, they cannot be changed per device.

# The host backend: `KernelAbstractions.CPU` on KernelAbstractions 0.9, its PoCL backend on 0.10.
const HOST_BACKEND = get_backend(Int[])
const HostBackend = typeof(HOST_BACKEND)

"""
    runs_threads(backend)

Whether `backend` is the host backend, whose arrays the `CPUThreads` algorithms can process on
Julia threads.
"""
runs_threads(::Backend) = false
runs_threads(::HostBackend) = true

"""
    runs_kernels(backend)

Whether AK's kernels (`@kernel cpu=false`) run on `backend`: every GPU backend, and the host
backend of KernelAbstractions 0.10, which compiles kernels for PoCL. KernelAbstractions 0.9's
`CPU` backend cannot run them.
"""
runs_kernels(::Backend) = true
runs_kernels(::HostBackend) = nameof(HostBackend) !== :CPU


# Checks shared by the algorithm families

# Algorithm names in error messages
algname(a) = nameof(typeof(a))
algname(::CPUThreads.SampleSort) = "CPUThreads.SampleSort"
algname(::CPUThreads.Partitioned) = "CPUThreads.Partitioned"

# Domain checks of the fields an algorithm was given explicitly, before any arithmetic uses them
check_fields(::Algorithm) = nothing

function check_positive(a, field)
    x = getfield(a, field)
    x === nothing || x >= 1 ||
        throw(ArgumentError("$(algname(a)): `$field` must be positive, got $x"))
    nothing
end

function check_pow2(a, field)
    x = getfield(a, field)
    x === nothing || (x >= 1 && ispow2(x)) ||
        throw(ArgumentError("$(algname(a)): `$field` must be a positive power of two, got $x"))
    nothing
end

function check_threads(a)
    check_positive(a, :max_tasks)
    check_positive(a, :min_elems)
end

check_fields(a::CPUThreads.Partitioned) = check_threads(a)

# Unset fields of a threaded algorithm: all threads, and the tuning's minimum per task
fill_threads(a::A, t) where {A} =
    A(something(a.max_tasks, Threads.nthreads()), something(a.min_elems, t.threads_min_elems))

function require_kernels(a, backend)
    runs_kernels(backend) || throw(ArgumentError(
        "$(algname(a)) runs AcceleratedKernels' GPU kernels, which " *
        "$(backend_name(backend)) cannot run (on the host, this needs KernelAbstractions 0.10); " *
        "use `Auto()` or a `CPUThreads` algorithm"))
    nothing
end

function require_threads(a, backend)
    runs_threads(backend) || throw(ArgumentError(
        "$(algname(a)) only runs on the host backend, not on $(backend_name(backend))"))
    nothing
end

# The settings of `foreachindex_on` for the loops of an operation that resolved to `a`
launch_kwargs(a::CPUThreads.Partitioned) = (; max_tasks=a.max_tasks, min_elems=a.min_elems)
launch_kwargs(a::Algorithm) = (; block_size=a.block_size)


# Backend resolution

backend_name(b::Backend) = nameof(typeof(b))

# Values that never determine the backend: lazy index collections (and Base's wrappers of them),
# scalars, and any other value that is not an array. Every other array votes with `get_backend`,
# which throws for array types that do not implement it.
backend_vote(_) = nothing
backend_vote(x::AbstractArray) = get_backend(x)
backend_vote(::Union{AbstractRange, CartesianIndices, LinearIndices}) = nothing
# Base's views, reshapes and permutations vote like the array they wrap, so that e.g. a reshaped
# range does not determine the backend either
backend_vote(x::Union{SubArray, Base.ReshapedArray, PermutedDimsArray}) = backend_vote(parent(x))
backend_vote(x::Tuple) = backend_votes(x...)
backend_vote(bc::Base.Broadcast.Broadcasted) = backend_votes(bc.args...)
backend_vote(x::Base.Broadcast.Extruded) = backend_vote(x.x)

backend_votes() = nothing
backend_votes(x, xs...) = merge_backends(backend_vote(x), backend_votes(xs...))

merge_backends(::Nothing, ::Nothing) = nothing
merge_backends(a, ::Nothing) = a
merge_backends(::Nothing, b) = b
function merge_backends(a, b)
    a == b || throw(ArgumentError(
        "the arguments live on different backends ($(backend_name(a)) and " *
        "$(backend_name(b))); pass `backend` explicitly if they are accessible from one of them"))
    a
end

# Lazy index collections, and Base's views, reshapes and permutations of them, have no backend
# (unlike `backend_vote`, this never asks an array for its backend, which an explicit `backend`
# makes unnecessary)
backendless(_) = false
backendless(::Union{AbstractRange, CartesianIndices, LinearIndices}) = true
backendless(x::Union{SubArray, Base.ReshapedArray, PermutedDimsArray}) = backendless(parent(x))

# An array for an operation's result: `similar(v, ...)`, unless `v` has no backend (a range, for
# instance), which gives a new array on `backend`
similar_on(backend, v, ::Type{T}=eltype(v), dims=size(v)) where {T} =
    backendless(v) ? KernelAbstractions.allocate(backend, T, dims) : similar(v, T, dims)
# A copy of `v`, likewise
copy_on(backend, v) = backendless(v) ? copyto!(similar_on(backend, v), collect(v)) : copy(v)

# `A` with axes starting at 1, as the kernels and threaded loops index it, for an `A` whose axes
# start elsewhere (an `OffsetArray`, for instance), and likewise the arrays of a `Broadcasted`
# object. The operations pass these on, and keep `A` for their checks.
one_based(A::AbstractArray) = Base.has_offset_axes(A) ? one_based_view(A) : A
function one_based(bc::Base.Broadcast.Broadcasted)
    Base.has_offset_axes(bc) || return bc
    args = Base.map(one_based, bc.args)
    # (keeping the shape of explicit axes, which may be larger than the arguments')
    return Base.Broadcast.Broadcasted(bc.f, args, Base.map(ax -> Base.OneTo(length(ax)), axes(bc)))
end
one_based(x) = x
# (a reshape keeps the linear indexing of an array of several dimensions, which starts at 1, and
# `OffsetArrays` makes it the parent array; Base does not reshape vectors with offset axes)
one_based_view(A::AbstractVector) = @view A[begin:end]
one_based_view(A::AbstractArray) = reshape(A, size(A))

# Whether `A` is a view `one_based` made. Base's `copyto!` and `fill!` index those element by
# element on the host, which GPU arrays disallow, so `copyto_on!` and `fill_on!` use a kernel.
has_offset_parent(A) =
    A isa Union{SubArray, Base.ReshapedArray} && Base.has_offset_axes(parent(A))

# `copyto!(dst, src)` of arrays of the same size
function copyto_on!(dst, src, backend)
    has_offset_parent(dst) || has_offset_parent(src) || return copyto!(dst, src)
    foreachindex_on(Base.OneTo(length(dst)), backend) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function fill_on!(A, x, backend)
    has_offset_parent(A) || return fill!(A, x)
    foreachindex_on(Base.OneTo(length(A)), backend) do i
        @inbounds A[i] = x
    end
    return A
end

"""
    resolve_backend(backend, args...)

The backend an operation runs on: `backend` if given, else the one every array in `args`
(destination first) agrees on, recursing into `Broadcasted` trees; ranges, `CartesianIndices`,
`LinearIndices`, Base's views, reshapes and permutations of them, and non-array values do not
count. Arguments on different backends are an `ArgumentError`. If no argument determines it, the
host backend is used.
"""
resolve_backend(backend::Backend, args...) = backend
function resolve_backend(::Nothing, args...)
    b = backend_votes(args...)
    b === nothing ? HOST_BACKEND : b
end
