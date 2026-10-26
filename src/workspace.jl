# Scratch memory: every operation that needs scratch plans it with `plan`, which resolves the
# algorithm and lists its buffers. The operation allocates the buffers, or takes them from a
# `Workspace` made from the same plan by `workspace`.


# A buffer requirement, with its element type in the type domain so that the allocated buffers,
# and the results computed with them, infer
struct Scratch{T, N}
    dims::NTuple{N, Int}
end
scratch(::Type{T}, dims::Integer...) where {T} = Scratch{T, length(dims)}(Int.(dims))
# (a buffer that is not `needed` for this call keeps its place without elements, so that an
# operation lists the same buffers for every length, and its plan infers)
scratch(::Type{T}, dims::Tuple; needed::Bool=true) where {T} =
    Scratch{T, length(dims)}(needed ? Int.(dims) : Base.map(_ -> 0, dims))

# An operation's scratch: its backend, its resolved algorithm (which the operation runs with),
# and its buffers, as a `NamedTuple` of `Scratch` requirements and of the plans of the
# operations it calls
struct Plan{B <: Backend, A, S <: NamedTuple}
    backend::B
    alg::A
    sizes::S
end

# `plan(op, args...; kwargs...)` has one method per operation that takes a workspace, with the
# operation's arguments (without `workspace`)
plan(op, args...; kwargs...) = throw(ArgumentError(
    "`$op` takes no workspace for arguments of types $(join(Base.map(Core.Typeof, args), ", "))"))

# The public form of the sizes: `(eltype, dims)` pairs, nested for the operations a plan calls
size_pairs(sizes::NamedTuple) = Base.map(size_pairs, sizes)
size_pairs(p::Plan) = size_pairs(p.sizes)
size_pairs(b::Scratch{T}) where {T} = (T, b.dims)
Base.show(io::IO, b::Scratch) = show(io, size_pairs(b))

nbytes(sizes::NamedTuple) = Base.sum(nbytes, values(sizes); init=0)
nbytes(p::Plan) = nbytes(p.sizes)
nbytes(b::Scratch{T}) where {T} = Base.prod(b.dims; init=1) * Base.elsize(Array{T})

# Whether a workspace made from plan `a` serves a call with plan `b`: the same backend,
# algorithms and buffer sizes, also in the plans of the operations they call. A plan holds
# nothing of the call's arrays, so that a workspace serves any call that needs the same buffers.
same_plan(a::Plan, b::Plan) =
    a.backend == b.backend && a.alg == b.alg && same_plan(a.sizes, b.sizes)
same_plan(a::NamedTuple{K}, b::NamedTuple{K}) where {K} = Base.all(Base.map(same_plan, a, b))
same_plan(a::Scratch, b::Scratch) = a == b
same_plan(a, b) = false

# The resolved algorithms of a plan, with those of the operations it calls
function algorithms(p::Plan)
    nested = (; (k => algorithms(v) for (k, v) in pairs(p.sizes) if v isa Plan)...)
    return isempty(nested) ? p.alg : (p.alg, nested)
end


"""
    Workspace

Scratch memory for one operation, made by [`workspace`](@ref) and passed to the operation with
its `workspace` keyword. A workspace records the backend and device it was made for, the
algorithms the operation and the operations it calls resolved to, and its buffers; a call whose
arguments need a different algorithm or other buffers throws an `ArgumentError` instead of using
it.

A workspace holds no state between calls, so it can be reused by any number of calls, but not by
calls that may run at the same time (on several tasks or streams). It covers the operations'
device memory, with two exceptions: on the host, the threaded algorithms still allocate small
per-task bookkeeping, and `Base.sort!` its own scratch for the per-task sorts of
`CPUThreads.SampleSort` and for the slices of a sort along `dims`; and before Julia 1.12, a
reduction of several arrays or of a `Broadcasted` object materializes it first.
"""
struct Workspace{P <: Plan, T <: NamedTuple}
    plan::P
    device::Int
    buffers::T
end

function Base.show(io::IO, ws::Workspace)
    print(io, "Workspace(", backend_name(ws.plan.backend), ", ", algname(ws.plan.alg), ", ",
          Base.format_bytes(nbytes(ws.plan)), ")")
end



"""
    workspace_size(op, args...; kwargs...) -> NamedTuple

The scratch buffers the call `op(args...; kwargs...)` needs, as a `NamedTuple` of
`(eltype, dims)` pairs (nested for the operations it calls), without allocating anything.
An operation with an algorithm whose call needs no scratch gives an empty `NamedTuple`; an
operation that takes no `workspace` keyword is an `ArgumentError`.

```julia
julia> AK.workspace_size(AK.sort!, CuArray(rand(Float32, 10_000)); alg=AK.RadixSort())
(temp = (Float32, (10000,)), hist = (UInt32, (5120,)), scan = (prefixes = (UInt32, (3,)),),
 key_range = (partials = (Tuple{UInt32, UInt32}, (40,)),))

julia> AK.workspace_size(AK.sum, CuArray(rand(Float32, 10_000)))
(partials = (Float32, (4,)),)
```
"""
workspace_size(op, args...; kwargs...) = size_pairs(plan(op, args...; kwargs...))

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
    p = plan(op, args...; kwargs...)
    return Workspace(p, KernelAbstractions.device(p.backend), allocate_scratch(p.backend, p))
end

allocate_scratch(backend, sizes::NamedTuple) = Base.map(s -> allocate_scratch(backend, s), sizes)
allocate_scratch(backend, p::Plan) = allocate_scratch(backend, p.sizes)
allocate_scratch(backend, b::Scratch{T}) where {T} = KernelAbstractions.allocate(backend, T, b.dims)

# The buffers of the plan: the workspace's, checked against the plan and the operation's arrays,
# or new ones
scratch_buffers(p::Plan, ::Nothing, arrays...) = allocate_scratch(p.backend, p)
function scratch_buffers(p::Plan, ws::Workspace, arrays...)
    ws.device == KernelAbstractions.device(p.backend) && same_plan(ws.plan, p) ||
        throw(plan_mismatch(ws, p))
    for b in leaves(ws.buffers), a in arrays
        mayalias(b, a) &&
            throw(ArgumentError("the workspace must not alias the operation's arrays"))
    end
    return ws.buffers
end
function plan_mismatch(ws::Workspace, p::Plan)
    ws.plan.backend == p.backend && ws.device == KernelAbstractions.device(p.backend) ||
        return ArgumentError("the workspace was made for another backend or device")
    algorithms(ws.plan) == algorithms(p) || return ArgumentError(
        "the workspace was made for $(algorithms(ws.plan)), but this call resolves to " *
        "$(algorithms(p))")
    return ArgumentError("the workspace holds the buffers $(size_pairs(ws.plan)), but this " *
                         "call needs $(size_pairs(p))")
end
scratch_buffers(p::Plan, ws, arrays...) =
    throw(ArgumentError("`workspace` must be a `Workspace` from `workspace`, got $(typeof(ws))"))

# Whether the buffer `b` may alias the array or `Broadcasted` source `a`
mayalias(b, a::AbstractArray) = Base.mightalias(b, a)
mayalias(b, a::Base.Broadcast.Broadcasted) = Base.any(x -> mayalias(b, x), a.args)
mayalias(b, a::Base.Broadcast.Extruded) = mayalias(b, a.x)
mayalias(b, a) = false

leaves(buffers::NamedTuple) = Iterators.flatten(Base.map(leaves, values(buffers)))
leaves(buffer::AbstractArray) = (buffer,)
