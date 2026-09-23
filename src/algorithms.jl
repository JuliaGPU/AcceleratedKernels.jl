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
    AcceleratedKernels.CPUThreads

Algorithms that run on Julia threads, for arrays on the host backend.
[`Auto`](@ref AcceleratedKernels.Auto) chooses them for host arrays; they are rejected for any
other backend.
"""
module CPUThreads

import ..AcceleratedKernels: SortAlgorithm

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

end # module CPUThreads


# Capabilities: correctness facts about a backend, checked for `Auto` and explicit algorithms
# alike. Unlike tunings, they cannot be changed per device.

# The host backend: `KernelAbstractions.CPU` on KernelAbstractions 0.9, its PoCL backend on 0.10.
const HOST_BACKEND = get_backend(Int[])
const HostBackend = typeof(HOST_BACKEND)

"""
    _runs_threads(backend)

Whether `backend` is the host backend, whose arrays the `CPUThreads` algorithms can process on
Julia threads.
"""
_runs_threads(::Backend) = false
_runs_threads(::HostBackend) = true

"""
    _runs_kernels(backend)

Whether AK's kernels (`@kernel cpu=false`) run on `backend`: every GPU backend, and the host
backend of KernelAbstractions 0.10, which compiles kernels for PoCL. KernelAbstractions 0.9's
`CPU` backend cannot run them.
"""
_runs_kernels(::Backend) = true
_runs_kernels(::HostBackend) = nameof(HostBackend) !== :CPU


# Backend resolution

_backend_name(b::Backend) = nameof(typeof(b))

# Values that never determine the backend: lazy index collections, scalars, and any other value
# that is not an array. Every other array votes with `get_backend`, which throws for array types
# that do not implement it.
_backend_vote(_) = nothing
_backend_vote(x::AbstractArray) = get_backend(x)
_backend_vote(::Union{AbstractRange, CartesianIndices, LinearIndices}) = nothing
_backend_vote(x::Tuple) = _backend_votes(x...)
_backend_vote(bc::Base.Broadcast.Broadcasted) = _backend_votes(bc.args...)
_backend_vote(x::Base.Broadcast.Extruded) = _backend_vote(x.x)

_backend_votes() = nothing
_backend_votes(x, xs...) = _backend_merge(_backend_vote(x), _backend_votes(xs...))

_backend_merge(::Nothing, ::Nothing) = nothing
_backend_merge(a, ::Nothing) = a
_backend_merge(::Nothing, b) = b
function _backend_merge(a, b)
    a == b || throw(ArgumentError(
        "the arguments live on different backends ($(_backend_name(a)) and " *
        "$(_backend_name(b))); pass `backend` explicitly if they are accessible from one of them"))
    a
end

# Lazy index collections, and Base's views, reshapes and permutations of them, have no backend
# (unlike `_backend_vote`, this never asks an array for its backend, which an explicit `backend`
# makes unnecessary)
_backend_free(_) = false
_backend_free(::Union{AbstractRange, CartesianIndices, LinearIndices}) = true
_backend_free(x::Union{SubArray, Base.ReshapedArray, PermutedDimsArray}) = _backend_free(parent(x))

# An array for an operation's result: `similar(v, ...)`, unless `v` has no backend (a range, for
# instance), which gives a new array on `backend`
_similar(backend, v, ::Type{T}=eltype(v), dims=size(v)) where {T} =
    _backend_free(v) ? KernelAbstractions.allocate(backend, T, dims) : similar(v, T, dims)
# A copy of `v`, likewise
_copy(backend, v) = _backend_free(v) ? copyto!(_similar(backend, v), collect(v)) : copy(v)

"""
    _resolve_backend(backend, args...)

The backend an operation runs on: `backend` if given, else the one every array in `args`
(destination first) agrees on, recursing into `Broadcasted` trees; ranges, `CartesianIndices`,
`LinearIndices` and non-array values do not count. Arguments on different backends are an
`ArgumentError`. If no argument determines it, the host backend is used.
"""
_resolve_backend(backend::Backend, args...) = backend
function _resolve_backend(::Nothing, args...)
    b = _backend_votes(args...)
    b === nothing ? HOST_BACKEND : b
end
