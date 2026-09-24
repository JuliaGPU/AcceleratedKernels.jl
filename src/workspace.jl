# Scratch memory: every operation that needs scratch plans it with `_plan`, which resolves the
# algorithm and lists its buffers. The operation allocates the buffers, or takes them from a
# `Workspace` made from the same plan by `workspace`.


# A buffer requirement, with its element type in the type domain so that the allocated buffers,
# and the results computed with them, infer
struct _Buf{T, N}
    dims::NTuple{N, Int}
end
_buffer(::Type{T}, dims::Integer...) where {T} = _Buf{T, length(dims)}(Int.(dims))
_buffer(::Type{T}, dims::Tuple) where {T} = _Buf{T, length(dims)}(Int.(dims))
Base.show(io::IO, b::_Buf{T}) where {T} = print(io, (T, b.dims))

# The public form of the sizes: `(eltype, dims)` pairs
_public_sizes(sizes::NamedTuple) = Base.map(_public_sizes, sizes)
_public_sizes(b::_Buf{T}) where {T} = (T, b.dims)


"""
    Workspace

Scratch memory for one operation, made by [`workspace`](@ref) and passed to the operation with
its `workspace` keyword. A workspace records the backend and device it was made for, the
algorithm the operation resolved to, and its buffers; a call whose arguments need a different
algorithm or other buffers throws an `ArgumentError` instead of using it.

A workspace holds no state between calls, so it can be reused by any number of calls, but not by
calls that may run at the same time (on several tasks or streams). It covers the operations'
device memory, with two exceptions: on the host, the threaded algorithms still allocate small
per-task bookkeeping, and `Base.sort!` its own scratch for the per-task sorts of
`CPUThreads.SampleSort` and for the slices of a sort along `dims`; and before Julia 1.12, a
reduction of several arrays or of a `Broadcasted` object materializes it first.
"""
struct Workspace{B <: Backend, A, N <: NamedTuple, S <: NamedTuple, T <: NamedTuple}
    backend::B
    device::Int
    alg::A
    nested::N
    sizes::S
    buffers::T
end

function Base.show(io::IO, ws::Workspace)
    print(io, "Workspace(", _backend_name(ws.backend), ", ", _algname(ws.alg), ", ",
          Base.format_bytes(_workspace_bytes(ws.sizes)), ")")
end

_workspace_bytes(sizes::NamedTuple) = Base.sum(_workspace_bytes, values(sizes); init=0)
_workspace_bytes(b::_Buf{T}) where {T} = Base.prod(b.dims; init=1) * Base.elsize(Array{T})


# An operation's scratch: its backend, its resolved algorithm (which the operation runs with),
# the resolved algorithms of the operations it calls, and its buffers, as a `NamedTuple` of
# `(eltype, dims)` requirements, or of the `NamedTuple`s of nested operations.
struct _Plan{B <: Backend, A, N <: NamedTuple, S <: NamedTuple}
    backend::B
    alg::A
    nested::N
    sizes::S
end
_Plan(backend, alg, sizes::NamedTuple) = _Plan(backend, alg, (;), sizes)

# `_plan(op, args...; kwargs...)` has one method per operation, taking the operation's arguments
# (without `workspace`); operations without scratch have none.
function _plan end



"""
    workspace_size(op, args...; kwargs...) -> NamedTuple

The scratch buffers the call `op(args...; kwargs...)` needs, as a `NamedTuple` of
`(eltype, dims)` pairs (nested for the operations it calls), without allocating anything.
An operation with an algorithm whose call needs no scratch gives an empty `NamedTuple`; the launch
wrappers (`foreachindex`, `map!`, `reverse!`, the searches) take no workspace.

```julia
julia> AK.workspace_size(AK.sort!, CuArray(rand(Float32, 10_000)); alg=AK.RadixSort())
(temp = (Float32, (10000,)), hist = (UInt32, (5120,)), scan = (prefixes = (UInt32, (3,)),),
 key_range = (partials = (Tuple{UInt32, UInt32}, (40,)),))

julia> AK.workspace_size(AK.sum, CuArray(rand(Float32, 10_000)))
(partials = (Float32, (40,)),)
```
"""
workspace_size(op, args...; kwargs...) = _public_sizes(_plan(op, args...; kwargs...).sizes)

"""
    workspace(op, args...; kwargs...) -> Workspace

Allocate the scratch memory of the call `op(args...; kwargs...)`, to pass to that operation with
the same arguments, or others that resolve to the same algorithm and need the same buffers,
through its `workspace` keyword. The operation then allocates no scratch memory of its own; it
still allocates its result where it returns a new array (`sort`, `findall`, `mapreduce` along
`dims`, ...), and compiles kernels as usual.

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 1_000_000))
ws = AK.workspace(AK.sort!, v)
for _ in 1:10
    rand!(v)
    AK.sort!(v; workspace=ws)        # no scratch allocations
end
```

The workspace is checked on every call: a different backend or device, a different resolved
algorithm (`Auto()` may choose another one for another length), also for the operations it calls,
or different buffer sizes are an `ArgumentError`, and so is a workspace whose buffers alias the
operation's arrays.
"""
function workspace(op, args...; kwargs...)
    p = _plan(op, args...; kwargs...)
    return Workspace(p.backend, KernelAbstractions.device(p.backend), p.alg, p.nested, p.sizes,
                     _allocate(p.backend, p.sizes))
end

_allocate(backend, sizes::NamedTuple) = Base.map(s -> _allocate(backend, s), sizes)
_allocate(backend, b::_Buf{T}) where {T} = KernelAbstractions.allocate(backend, T, b.dims)

# The buffers of the plan: the workspace's, checked against the plan and the operation's arrays,
# or new ones
_buffers(p::_Plan, ::Nothing, arrays...) = _allocate(p.backend, p.sizes)
function _buffers(p::_Plan, ws::Workspace, arrays...)
    ws.backend == p.backend && ws.device == KernelAbstractions.device(p.backend) ||
        throw(ArgumentError("the workspace was made for another backend or device"))
    ws.alg == p.alg && ws.nested == p.nested || throw(ArgumentError(
        "the workspace was made for $(ws.alg) $(ws.nested), but this call resolves to " *
        "$(p.alg) $(p.nested)"))
    ws.sizes == p.sizes || throw(ArgumentError(
        "the workspace's buffers $(ws.sizes) differ from those this call needs, $(p.sizes)"))
    for b in _leaves(ws.buffers), a in arrays
        _aliases(b, a) &&
            throw(ArgumentError("the workspace must not alias the operation's arrays"))
    end
    return ws.buffers
end

# Whether the buffer `b` may alias the array or `Broadcasted` source `a`
_aliases(b, a::AbstractArray) = Base.mightalias(b, a)
_aliases(b, a::Base.Broadcast.Broadcasted) = Base.any(x -> _aliases(b, x), a.args)
_aliases(b, a::Base.Broadcast.Extruded) = _aliases(b, a.x)
_aliases(b, a) = false
_buffers(p::_Plan, ws, arrays...) =
    throw(ArgumentError("`workspace` must be a `Workspace` from `workspace`, got $(typeof(ws))"))

_leaves(buffers::NamedTuple) = Iterators.flatten(Base.map(_leaves, values(buffers)))
_leaves(buffer::AbstractArray) = (buffer,)
