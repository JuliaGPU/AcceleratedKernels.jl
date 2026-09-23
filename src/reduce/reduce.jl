# Backend implementations
include("utilities.jl")
include("tuning.jl")

const MapReduceSource = Union{AbstractArray, Base.Broadcast.Broadcasted}

function _mapreduce_check_map_axes(src::AbstractArray, srcs::AbstractArray...)
    src_axes = axes(src)
    for other in srcs
        axes(other) == src_axes || throw(DimensionMismatch("all input arrays must have the same axes"))
    end
    return nothing
end

include("mapreduce_1d_cpu.jl")
include("mapreduce_1d_gpu.jl")
include("mapreduce_nd.jl")


"""
    reduce(
        op, src::AbstractArray;
        backend=nothing,
        init,
        neutral=neutral_element(op, typeof(init)),
        dims=nothing,
        alg::Algorithm=Auto(),
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Reduce `src` along dimensions `dims` using the binary operator `op`, which must be associative
and commutative. If `dims` is `nothing` or `:`, reduce `src` to a scalar, returned on the host.
If `dims` is an integer or a collection of integers, reduce `src` along those dimension(s). The
`init` value is used as the initial value for the reduction; `neutral` is the neutral element for
the operator `op`.

The returned type is the same as `init` - to control output precision, specify `init` explicitly.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host and [`BlockReduce`](@ref) on GPUs, with the
device's settings. `backend` is derived from `src`.

On the host, multithreading reductions only improves performance for operations that hide the
memory latency and thread launch overhead, e.g. reductions of tuples or structs, or expensive
operators.

The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
(`dims=nothing` or `dims=:`), `length(temp) >= 2 * cld(length(src), items_per_thread * block_size)`
of the `BlockReduce` settings is required. For reduction along dimensions (`dims` is an integer or
a collection of integers), `temp` is used as the destination array, and thus must have the exact
dimensions required - i.e. same dimensionwise sizes as `src`, except for the reduced dimension(s)
which become 1; there are some corner cases when one dimension is zero, check against
`Base.reduce` for CPU arrays for exact behavior.

# Examples
Computing a sum, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsum = AK.reduce((x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums in a 2D matrix, with explicit settings:
```julia
import AcceleratedKernels as AK
using Metal

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsum = AK.reduce(+, m; init=zero(eltype(m)), dims=1)
mcolsum = AK.reduce(+, m; init=zero(eltype(m)), dims=2, alg=AK.BlockReduce(block_size=512))
```
"""
reduce(op, src::AbstractArray; kwargs...) = mapreduce(identity, op, src; kwargs...)


"""
    mapreduce(
        f, op, src::AbstractArray, srcs::AbstractArray...;
        backend=nothing,
        init,
        neutral=neutral_element(op, typeof(init)),
        dims=nothing,
        alg::Algorithm=Auto(),
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Reduce `src` along dimensions `dims` using the binary operator `op` after applying `f`
elementwise. The keywords are those of [`reduce`](@ref); `init` is used as the initial value for
the reduction (i.e. after mapping), and `neutral` is needed for an efficient GPU implementation
that also allows a nonzero `init`.

Multiple input arrays are supported with the same axes. This follows `Base.mapreduce(f, op, A,
B, ...)` semantics: `f` is mapped across corresponding elements of the inputs and the mapped
values are reduced without materializing the intermediate array. Mismatched axes throw
`DimensionMismatch`. `backend` is derived from all inputs. A `Broadcasted` object is also
accepted as the single source, with singleton-expanding broadcast semantics, for array backends.

# Examples
Computing a sum of squares, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsumsq = AK.mapreduce(x -> x * x, (x, y) -> x + y, v; init=zero(eltype(v)))
```

Computing dimensionwise sums of squares in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

f(x) = x * x
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsumsq = AK.mapreduce(f, +, m; init=zero(eltype(m)), dims=1)
```

Computing a two-input dimensional reduction:
```julia
rows = AK.mapreduce((x, y) -> x * y, +, a, b; init=0f0, dims=1)
```
"""
function mapreduce(
    f, op, src::MapReduceSource, srcs::AbstractArray...;
    backend::Union{Nothing, Backend}=nothing,
    init,
    neutral=neutral_element(op, typeof(init)),
    dims=nothing,
    alg::Algorithm=Auto(),
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if !isempty(srcs)
        src isa AbstractArray ||
            throw(ArgumentError("a Broadcasted source cannot be combined with more arrays"))
        _mapreduce_check_map_axes(src, srcs...)
        src = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, src, srcs...))
        f = identity
    end
    backend = _resolve_backend(backend, src, temp)
    a = _resolve_reduce(alg, backend, typeof(init), dims)
    _mapreduce_impl(f, op, src, backend, a; init, neutral, dims, temp)
end


function _mapreduce_impl(
    f, op, src::MapReduceSource, backend::Backend, alg;
    init, neutral, dims, temp,
)
    # scalar *linear* indexing into a multidimensional Broadcasted object is
    # only available on Julia 1.12; on earlier version, materialize it first.
    if VERSION < v"1.12-" && src isa Base.Broadcast.Broadcasted
        src = Base.Broadcast.materialize(src)
    end

    if _whole(dims)
        if alg isa BlockReduce
            mapreduce_1d_gpu(
                f, op, src, backend;
                init, neutral,
                block_size=alg.block_size, items_per_thread=alg.items_per_thread,
                temp, switch_below=alg.switch_below,
            )
        else
            mapreduce_1d_cpu(
                f, op, src, backend;
                init, neutral,
                max_tasks=alg.max_tasks, min_elems=alg.min_elems,
            )
        end
    else
        mapreduce_nd(f, op, src, backend, alg; init, neutral, dims, temp)
    end
end
