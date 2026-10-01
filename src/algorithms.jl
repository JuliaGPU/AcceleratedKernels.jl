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

Supertype of the scan (`accumulate`) algorithms: [`ScanPrefixes`](@ref),
[`DecoupledLookback`](@ref) and [`SliceScan`](@ref). Scans also accept
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

"""
    _supports_lookback(backend)

Whether [`DecoupledLookback`](@ref) is correct on `backend`: it needs a device-scope memory fence
(`_decoupled_fence`), atomic loads and stores of the block flags, and forward progress between
workgroups, since a block spins until an earlier one publishes its prefix. AK's extensions
declare it for the backends where all three are known to hold.
"""
_supports_lookback(::Backend) = false


# Checks shared by the algorithm families

# Algorithm names in error messages
_algname(a) = nameof(typeof(a))
_algname(::CPUThreads.SampleSort) = "CPUThreads.SampleSort"
_algname(::CPUThreads.Partitioned) = "CPUThreads.Partitioned"

# Domain checks of the fields an algorithm was given explicitly, before any arithmetic uses them
_checkdomain(::Algorithm) = nothing

function _check_positive(a, field)
    x = getfield(a, field)
    x === nothing || x >= 1 ||
        throw(ArgumentError("$(_algname(a)): `$field` must be positive, got $x"))
    nothing
end

function _check_pow2(a, field)
    x = getfield(a, field)
    x === nothing || (x >= 1 && ispow2(x)) ||
        throw(ArgumentError("$(_algname(a)): `$field` must be a positive power of two, got $x"))
    nothing
end

function _check_threads(a)
    _check_positive(a, :max_tasks)
    _check_positive(a, :min_elems)
end

_checkdomain(a::CPUThreads.Partitioned) = _check_threads(a)

# Unset fields of a threaded algorithm: all threads, and the tuning's minimum per task
_fill_threads(a::A, t) where {A} =
    A(something(a.max_tasks, Threads.nthreads()), something(a.min_elems, t.threads_min_elems))

function _require_kernels(a, backend)
    _runs_kernels(backend) || throw(ArgumentError(
        "$(_algname(a)) runs AcceleratedKernels' GPU kernels, which " *
        "$(_backend_name(backend)) cannot run (on the host, this needs KernelAbstractions 0.10); " *
        "use `Auto()` or a `CPUThreads` algorithm"))
    nothing
end

function _require_threads(a, backend)
    _runs_threads(backend) || throw(ArgumentError(
        "$(_algname(a)) only runs on the host backend, not on $(_backend_name(backend))"))
    nothing
end


# Backend resolution

_backend_name(b::Backend) = nameof(typeof(b))

# Values that never determine the backend: lazy index collections, scalars, and any other value
# that is not an array. Every other array votes with `get_backend`, which throws for array types
# that do not implement it.
_backend_vote(_) = nothing
_backend_vote(::Union{AbstractRange, CartesianIndices, LinearIndices}) = nothing
_backend_vote(x::Tuple) = _backend_votes(x...)
_backend_vote(bc::Base.Broadcast.Broadcasted) = _backend_votes(bc.args...)
_backend_vote(x::Base.Broadcast.Extruded) = _backend_vote(x.x)
# An array that wraps others (its `parent` is another array, or a tuple of them for a wrapper of
# several, like a mapped array of two arrays) votes like the arrays it wraps. KernelAbstractions'
# `get_backend` recurses through `parent` the same way, and is asked where it can answer, or
# where the wrapper defines its own method; it throws for a chain of parents ending in a range or
# a tuple, where a reshaped range, or a lazy array computed from ranges, has no backend here.
function _backend_vote(x::AbstractArray)
    p = parent(x)
    p === x && return get_backend(x)
    _wraps_storage(p) || _defines_backend(x) ? get_backend(x) : _backend_vote(p)
end

# Whether `x` is an array with memory, or a chain of single parents ending in one
_wraps_storage(::Union{AbstractRange, CartesianIndices, LinearIndices}) = false
_wraps_storage(x::AbstractArray) = (p = parent(x); p === x || _wraps_storage(p))
_wraps_storage(_) = false

# Whether `get_backend(x)` has a method other than KernelAbstractions' fallback for arrays
_defines_backend(x) =
    which(get_backend, Tuple{typeof(x)}) !== which(get_backend, Tuple{AbstractArray})

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

# Lazy index collections, and Base's views, reshapes and permutations of them, have no backend and
# can be evaluated on the host (unlike `_backend_vote`, this never asks an array for its backend,
# which an explicit `backend` makes unnecessary). Other lazy arrays may compute their elements
# from arrays they capture rather than wrap, so they are evaluated on the backend instead.
_backend_free(_) = false
_backend_free(::Union{AbstractRange, CartesianIndices, LinearIndices}) = true
_backend_free(x::Union{SubArray, Base.ReshapedArray, PermutedDimsArray}) = _backend_free(parent(x))

# The array that holds the memory behind `v`: `v` itself unless it wraps others, else the first of
# the arrays it wraps that holds memory; `nothing` if there is none (a range, or a lazy array over
# one)
_storage(v::Tuple) = _storage_first(v...)
function _storage(v::AbstractArray)
    _backend_free(v) && return nothing
    p = parent(v)
    p === v ? v : _storage(p)
end
_storage(_) = nothing
_storage_first() = nothing
_storage_first(x, xs...) = (s = _storage(x); s === nothing ? _storage_first(xs...) : s)

# Whether `v` wraps the memory `s` of another array type than a host `Array` but gets Base's
# `similar` fallback, a host `Array` (a lazy array without a `similar` method of its own, for
# instance). Checked on an empty result.
_similar_falls_back(v, s) =
    s !== v && !(s isa Array) && similar(v, eltype(v), Base.map(zero, size(v))) isa Array

# An array for an operation's result: `similar(v, ...)`, unless `v` holds no memory (a range, or a
# lazy array over one), or gets Base's fallback for its `similar`, which give a new array on
# `backend`
function _similar(backend, v, ::Type{T}=eltype(v), dims=size(v)) where {T}
    s = _storage(v)
    s === nothing || _similar_falls_back(v, s) ? KernelAbstractions.allocate(backend, T, dims) :
                                                  similar(v, T, dims)
end

# Whether `v` is copied with a kernel on `backend`. Its own array, Base's wrappers of arrays in host
# memory and GPUArrays' wrappers of its arrays are copied by Base or GPUArrays, as backend-free
# arrays are after collecting them on the host, and everything is on the host backend. Copies of
# other wrappers fall back to scalar indexing, and lazy arrays over ranges may read device arrays
# they capture.
function _copies_by_kernel(backend, v)
    (_runs_threads(backend) || _backend_free(v) || parent(v) === v || v isa AnyGPUArray) &&
        return false
    !(_storage(v) isa Union{Array, BitArray})
end

# Copy the elements of `src` into `dst`, which has as many, in linear order
function _kernel_copy!(backend, dst, src)
    d0, s0 = firstindex(dst), firstindex(src)
    _foreachindex(Base.OneTo(length(src)), backend) do i
        @inbounds dst[d0 + i - 1] = src[s0 + i - 1]
    end
    dst
end

# `copyto!(dst, src)`, `copy(v)` and `Array(v)` for any source on `backend`
_copyto!(backend, dst, src) =
    _copies_by_kernel(backend, src) ? _kernel_copy!(backend, dst, src) :
    copyto!(dst, _backend_free(src) ? collect(src) : src)
_copy(backend, v) =
    _copies_by_kernel(backend, v) ? _kernel_copy!(backend, _similar(backend, v), v) :
    _backend_free(v) ? copyto!(_similar(backend, v), collect(v)) : copy(v)
_host_array(backend, v) = _copies_by_kernel(backend, v) ? Array(_copy(backend, v)) : Array(v)

"""
    _resolve_backend(backend, args...)

The backend an operation runs on: `backend` if given, else the one every array in `args`
(destination first) agrees on, recursing into `Broadcasted` trees and into the arrays that
wrappers wrap; ranges, `CartesianIndices`, `LinearIndices`, wrappers of only those, and non-array
values do not count. Arguments on different backends are an `ArgumentError`. If no argument
determines it, the host backend is used.
"""
_resolve_backend(backend::Backend, args...) = backend
function _resolve_backend(::Nothing, args...)
    b = _backend_votes(args...)
    b === nothing ? HOST_BACKEND : b
end
