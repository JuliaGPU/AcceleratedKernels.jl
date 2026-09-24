# Backend implementations
include("utilities.jl")
include("tuning.jl")

const MapReduceSource = Union{AbstractArray, Base.Broadcast.Broadcasted}

# The size of a reduction source along `d`; `Broadcasted` objects only have `size` without `d`
_srcsize(src, d::Int) = d <= ndims(src) ? size(src)[d] : 1

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
    reduce(op, src::AbstractArray; kwargs...)

Reduce `src` with the binary operator `op`, which must be associative and commutative; the
keywords are those of [`mapreduce`](@ref). Equivalent to `mapreduce(identity, op, src; kwargs...)`.

# Examples
```julia
import AcceleratedKernels as AK
using Metal

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
AK.reduce(+, m)                                     # a host scalar
AK.reduce(max, m; dims=1)                           # a 1×100_000 MtlArray
AK.reduce(+, m; dims=2, alg=AK.BlockReduce(block_size=512))
```
"""
reduce(op, src::AbstractArray; kwargs...) = mapreduce(identity, op, src; kwargs...)


"""
    mapreduce(
        f, op, src, srcs::AbstractArray...;
        backend=nothing,
        init=<none>,
        neutral=nothing,
        acctype=nothing,
        dims=:,
        alg::Algorithm=Auto(),
        workspace=nothing,
    )

Apply `f` to each element of `src` and reduce the results with the binary operator `op`. With
`dims=:` (the default; `nothing` is accepted too) the result is a scalar, returned on the host.
With `dims` an integer or a collection of integers, the result is a new array on `src`'s backend
with size 1 along those dimensions; [`mapreducedim!`](@ref) reduces into an existing array
instead, and states the contract all reductions follow. In short:

- `op` must be associative and commutative.
- `init`, when given, is applied exactly once, as `op(init, partial)`, and is the result of an
  empty reduction. Without `init`, an empty reduction is an `ArgumentError`: of a whole array, and
  along `dims` of any output whose slice is empty. ([`sum`](@ref), [`prod`](@ref) and
  [`count`](@ref) give zero or one instead.)
- The result has the accumulator type: `acctype` when it is given, else the type the fold of `op`
  settles on from `init`'s type (when given) and the mapped elements; `op(init, partial)` is
  converted to it. So `AK.sum(Int8[1, 2])` is an `Int`, and `AK.sum(Int8[1 2]; dims=1,
  init=Int16(0))` an array of `Int`s. A single element has that type too:
  `AK.reduce((a, b) -> a + b, [true]) === 1`. (An empty reduction returns `init` as it is.)
- `neutral`, a two-sided identity of `op`, only seeds partial results and never appears in the
  result. It defaults to `GPUArraysCore.neutral_element(op, T)` where that is defined; for other
  operators, each partial result starts from its first element instead.

These rules are AcceleratedKernels' own, and differ from Base's in a few places, e.g. for empty
reductions without `init`; see [Differences from Base](@ref).

Several source arrays with equal axes are reduced without materializing the mapped array
(`f` takes one argument per array); mismatched axes throw a `DimensionMismatch`. A `Broadcasted`
object is also accepted as the single source. Before Julia 1.12, both are materialized when the
reduction runs. `backend` is derived from all sources.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host and [`BlockReduce`](@ref) on GPUs, with the
device's settings. On the host, multithreading reductions only improves performance for
operations that hide the memory latency and thread launch overhead, e.g. reductions of tuples or
structs, or expensive operators.

`workspace` takes the scratch memory of a [`workspace`](@ref) made for the same call, so that
the reduction allocates none of its own (it still allocates its result along `dims`, and before
Julia 1.12 a materialized source).

# Examples
Computing a sum of squares, reducing down to a scalar that is copied to host:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray{Int16}(rand(1:1000, 100_000))
vsumsq = AK.mapreduce(x -> Int(x) * x, +, v)
```

Computing dimensionwise sums of squares in a 2D matrix:
```julia
import AcceleratedKernels as AK
using Metal

f(x) = x * x
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
mrowsumsq = AK.mapreduce(f, +, m; dims=1)
```

Computing a two-input dimensional reduction:
```julia
rows = AK.mapreduce((x, y) -> x * y, +, a, b; dims=1)
```
"""
function mapreduce(
    f, op, src::MapReduceSource, srcs::AbstractArray...;
    backend::Union{Nothing, Backend}=nothing,
    init=_NoInit(),
    neutral=nothing,
    acctype=nothing,
    dims=:,
    alg::Algorithm=Auto(),
    workspace=nothing,
)
    f, src = _mapreduce_fuse(f, src, srcs)
    s = _mapreduce_setup(f, op, src, backend, init, neutral, acctype, dims, alg)
    bufs = _buffers(s.plan, workspace, src)
    return _mapreduce_run(f, op, _mapreduce_source(src), s, init, bufs)
end

function _plan(
    ::typeof(mapreduce), f, op, src::MapReduceSource, srcs::AbstractArray...;
    backend=nothing, init=_NoInit(), neutral=nothing, acctype=nothing, dims=:,
    alg::Algorithm=Auto(),
)
    f, src = _mapreduce_fuse(f, src, srcs)
    return _mapreduce_setup(f, op, src, backend, init, neutral, acctype, dims, alg).plan
end

_plan(::typeof(reduce), op, src::AbstractArray; kwargs...) =
    _plan(mapreduce, identity, op, src; kwargs...)

# The source of a reduction: several arrays become one `Broadcasted` object
_mapreduce_fuse(f, src, ::Tuple{}) = (f, src)
function _mapreduce_fuse(f, src, srcs::Tuple)
    src isa AbstractArray ||
        throw(ArgumentError("a Broadcasted source cannot be combined with more arrays"))
    _mapreduce_check_map_axes(src, srcs...)
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, src, srcs...))
    return identity, bc
end
# Scalar *linear* indexing into a multidimensional `Broadcasted` object is only available on
# Julia 1.12, so earlier versions materialize it when the reduction runs (after the plan and the
# workspace checks, which see the original arrays); the plan treats it as a dense array.
_mapreduce_source(src) = src
_mapreduce_source(src::Base.Broadcast.Broadcasted) =
    VERSION < v"1.12-" ? _materialize_source(src) : src

# The seed of the partial results (`_reduce_seed`), where an accumulator type exists
_mapreduce_seed(op, ::Type{A}, neutral) where {A} =
    A === Union{} ? nothing : _reduce_seed(op, A, neutral)

# Everything a reduction resolves before touching data: its backend, algorithm and scratch (the
# plan), accumulator type and partial-result seed, and for reductions along `dims` the result
function _mapreduce_setup(f, op, src, backend, init, neutral, acctype, dims, alg)
    backend = _resolve_backend(backend, src)
    M = _mapped_eltype(f, src)
    # The accumulator type, from `init`'s type
    A = _acctype(op, init isa _NoInit ? Union{} : typeof(init), M, acctype)

    if _whole(dims)
        a = _resolve_reduce(alg, backend, A, dims)
        seed = _mapreduce_seed(op, A, neutral)
        # (the buffer's name does not depend on the length, so that results infer)
        sizes = if a isa BlockReduce && seed !== nothing
            n = _mapreduce_1d_partials(length(src), a.block_size, a.items_per_thread,
                                       a.switch_below)
            (; partials=_buffer(typeof(seed), n))
        else
            (;)
        end
        return (; plan=_Plan(backend, a, sizes), A=Val(A), seed)
    end

    dims_valid = _reduced_dims(dims, ndims(src))
    # The result has the accumulator type; where there is none (`op` always throws), only an
    # empty reduction can succeed
    R_type = A !== Union{} ? A : !(init isa _NoInit) ? typeof(init) : M === Union{} ? Nothing : M
    dst_sizes = ntuple(d -> d in dims_valid ? 1 : _srcsize(src, d), ndims(src))
    a = _resolve_reduce(alg, backend, A, dims_valid)
    seed = _mapreduce_seed(op, A, neutral)
    sizes = _mapreduce_nd_sizes(src, backend, a, A, seed, dims_valid, Base.prod(dst_sizes))
    return (; plan=_Plan(backend, a, sizes), A=Val(A), seed, dims_valid, R_type=Val(R_type),
            dst_sizes)
end

# Run the reduction set up by `_mapreduce_setup`, with the plan's scratch buffers `bufs`
function _mapreduce_run(f, op, src, s, init, bufs)
    backend, a = s.plan.backend, s.plan.alg
    if !haskey(s, :dims_valid)
        return _mapreduce_whole(f, op, src, backend, a, _unval(s.A); init, neutral=s.seed,
                                partials=get(bufs, :partials, nothing))
    end
    if init isa _NoInit && Base.any(d -> _srcsize(src, d) == 0, s.dims_valid) &&
       Base.prod(s.dst_sizes) > 0
        throw(ArgumentError(
            "reducing over an empty dimension is not allowed without `init`; pass `init`"))
    end
    dst = KernelAbstractions.allocate(backend, _unval(s.R_type), s.dst_sizes)
    return mapreduce_nd!(f, op, dst, src, backend, a, _unval(s.A);
                         init, neutral=s.seed, dims_valid=s.dims_valid, bufs)
end

# A reduction nested in another operation, with the scratch buffers of its plan in the outer one
_mapreduce_nested(f, op, src, bufs; backend, init=_NoInit(), neutral=nothing, dims=:, alg) =
    _mapreduce_run(f, op, src,
                   _mapreduce_setup(f, op, src, backend, init, neutral, nothing, dims, alg),
                   init, bufs)


"""
    mapreducedim!(
        f, op, R::AbstractArray, src;
        backend=get_backend(R),
        init=<none>,
        neutral=nothing,
        overwrite::Bool=false,
        acctype=nothing,
        alg::Algorithm=Auto(),
        workspace=nothing,
    ) -> R

Reduce `src` (an array or a `Broadcasted` object) into `R`: the dimensions along which `R` has
size 1 and `src` does not are reduced, and the others must match (`R` may leave off trailing
dimensions of size 1). `R` must not alias `src`.

This is the contract of every AcceleratedKernels reduction, modelled on CUB's rather than on
Base's (see [Differences from Base](@ref)):

- **Algebra.** `op` must be associative and commutative. Like every GPU reduction, AK combines
  elements in an order that depends on the algorithm, its settings and the shape, not in element
  order; for a given setting and shape the order is fixed, so results are reproducible on one
  device. Non-commutative operators such as string concatenation or matrix products are not
  supported.
- **Partial results** start from `neutral`, which must be a two-sided identity of `op` and never
  appears in the result. It defaults to `GPUArraysCore.neutral_element(op, T)` where that is
  defined; otherwise each partial result starts from its first element,
  `Base.mapreduce_first(f, op, x)`, so no neutral element is needed.
- **Accumulator type.** Partial results have one type, and are converted to `eltype(R)` only when
  stored. It is `acctype` when that is given; otherwise the type the fold of `op` settles on,
  starting from `eltype(R)` and the mapped elements. Every element is also a one-element partial
  result, and partial results are combined with each other, so those types are joined in as well
  (`R`'s values and `init` are applied once, and are not partial results). A sum of `Int8`s
  accumulates in `Int`, and `Float32`s reduced into a `Float64` array accumulate in `Float64`.
  An `acctype` that cannot hold the partial results at all is an `ArgumentError`; whether their
  values fit is the caller's obligation (a value that does not fit throws an `InexactError`,
  where the backend reports errors thrown in kernels).
- **Result.** For each output whose slice is not empty, with `partial` its reduction:
  `R[i] = op(init, partial)` when `init` is given (applied once); else `partial` with
  `overwrite=true`; else `op(R[i], partial)`, folding into `R`'s previous value as
  `Base.mapreducedim!` does.
- **Empty slices.** An output whose slice is empty is set to `init` when that is given, and is
  otherwise not written, with or without `overwrite`.
- **Errors.** Mismatched shapes are a `DimensionMismatch`. An `R` that aliases `src`, an `op`
  that inference shows always throws for these types (when there is something to combine; an
  explicit `acctype` that `op` cannot combine at all is rejected whatever the input), a
  non-bits accumulator type for a kernel algorithm, and an algorithm that cannot run on the
  backend are `ArgumentError`s.

`workspace` takes the scratch memory of a [`workspace`](@ref) made for the same call.

```julia
import AcceleratedKernels as AK
using CUDA

A = CuArray([1 3; 2 4])
R = CuArray([10 20])
AK.mapreducedim!(identity, +, R, A)                  # [13 27]
AK.mapreducedim!(identity, +, R, A; overwrite=true)  # [3 7]
AK.mapreducedim!(identity, +, R, A; init=100)        # [103 107]
```
"""
function mapreducedim!(
    f, op, R::AbstractArray, src::MapReduceSource;
    backend::Union{Nothing, Backend}=nothing,
    init=_NoInit(),
    neutral=nothing,
    overwrite::Bool=false,
    acctype=nothing,
    alg::Algorithm=Auto(),
    workspace=nothing,
)
    s = _mapreducedim_setup(f, op, R, src, backend, init, neutral, overwrite, acctype, alg)
    bufs = _buffers(s.plan, workspace, R, src)
    mapreduce_nd!(f, op, s.dst, _mapreduce_source(s.src), s.plan.backend, s.plan.alg, _unval(s.A);
                  init=s.init, neutral=s.seed, dims_valid=s.dims_valid, bufs)
    return R
end

_plan(::typeof(mapreducedim!), f, op, R::AbstractArray, src::MapReduceSource;
      backend=nothing, init=_NoInit(), neutral=nothing, overwrite::Bool=false, acctype=nothing,
      alg::Algorithm=Auto()) =
    _mapreducedim_setup(f, op, R, src, backend, init, neutral, overwrite, acctype, alg).plan

function _mapreducedim_setup(f, op, R, src, backend, init, neutral, overwrite, acctype, alg)
    backend = _resolve_backend(backend, R, src)
    nd = ndims(src)
    for d in 1:max(nd, ndims(R))
        sR, sA = size(R, d), _srcsize(src, d)
        sR == 1 || sR == sA || throw(DimensionMismatch(
            "cannot reduce an array of size $(size(src)) into one of size $(size(R))"))
    end
    _check_noalias(R, src)
    dst_sizes = ntuple(d -> size(R, d), nd)
    dst = size(R) == dst_sizes ? R : reshape(R, dst_sizes)
    dims_valid = Tuple(d for d in 1:nd if dst_sizes[d] == 1 && _srcsize(src, d) != 1)
    A = _acctype(op, eltype(dst), _mapped_eltype(f, src), acctype)
    a = _resolve_reduce(alg, backend, A, dims_valid)
    seed = _mapreduce_seed(op, A, neutral)
    sizes = _mapreduce_nd_sizes(src, backend, a, A, seed, dims_valid, length(dst))
    init = init isa _NoInit && !overwrite ? _Fold() : init
    return (; plan=_Plan(backend, a, sizes), src, dst, A=Val(A), seed, dims_valid, init)
end

function _check_noalias(R, src::AbstractArray)
    Base.mightalias(R, src) &&
        throw(ArgumentError("the destination of a reduction must not alias its source"))
    nothing
end
_check_noalias(R, src::Base.Broadcast.Broadcasted) = foreach(a -> _check_noalias(R, a), src.args)
_check_noalias(R, src::Base.Broadcast.Extruded) = _check_noalias(R, src.x)
_check_noalias(R, src) = nothing

# Reduce all of `src` to a host value, with accumulator type `A` and partial-result seed
# `neutral`; `partials` is the plan's scratch
function _mapreduce_whole(f, op, src, backend, alg, ::Type{A}; init, neutral, partials) where {A}
    if length(src) == 0
        init isa _NoInit || return init
        throw(ArgumentError(
            "reducing over an empty collection is not allowed without `init`; pass `init`"))
    end
    # (a single element without `init` has nothing to combine, even when `op` always throws)
    if A === Union{} && length(src) == 1 && init isa _NoInit
        return @allowscalar Base.mapreduce_first(f, op, src[first(eachindex(src))])
    end
    _check_acctype(op, f, A)

    # The result, `op(init, partial)` or `partial`, has the accumulator type
    return convert(A, _mapreduce_whole_run(f, op, src, backend, alg; init, neutral, partials))
end

function _mapreduce_whole_run(f, op, src, backend, alg; init, neutral, partials)
    if alg isa BlockReduce
        mapreduce_1d_gpu(
            f, op, src, backend;
            init, neutral,
            block_size=alg.block_size, items_per_thread=alg.items_per_thread,
            partials, switch_below=alg.switch_below,
        )
    else
        mapreduce_1d_cpu(
            f, op, src, backend;
            init, neutral,
            max_tasks=alg.max_tasks, min_elems=alg.min_elems,
        )
    end
end
