# Backend implementations
include("tuning.jl")

const MapReduceSource = Union{AbstractArray, Base.Broadcast.Broadcasted}

# The size of a reduction source along `d`; `Broadcasted` objects only have `size` without `d`
srcsize(src, d::Int) = d <= ndims(src) ? size(src)[d] : 1

# The reduced dimensions `dims` (an integer or a collection of integers), as Base validates them:
# duplicates are ignored, and dimensions beyond `ndims` reduce nothing.
function reduced_dims(dims, ndim::Int)
    dims_src = dims isa Number ? (dims,) : dims
    dims_buf = Int[]
    for d in dims_src
        d isa Integer || throw(ArgumentError("reduced dimension(s) must be integers"))
        dim = Int(d)
        dim < 1 && throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
        push!(dims_buf, dim)
    end
    return Tuple(d for d in Base.unique(dims_buf) if d <= ndim)
end

include("mapreduce_1d_cpu.jl")
include("mapreduce_ordered.jl")
include("mapreduce_1d_gpu.jl")
include("layout.jl")
include("shapes.jl")
include("mapreduce_nd.jl")
include("kernels_nd.jl")


"""
    reduce(op, src::AbstractArray; kwargs...)

Reduce `src` with the binary operator `op`, which must be associative, and commutative unless
`commutative=false`; the keywords are those of [`mapreduce`](@ref). Equivalent to `mapreduce(identity, op, src; kwargs...)`.

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
reduce(op::OP, src::AbstractArray; kwargs...) where {OP} = mapreduce(identity, op, src; kwargs...)


"""
    mapreduce(
        f, op, src, srcs::AbstractArray...;
        backend=nothing,
        init=<none>,
        neutral=nothing,
        acctype=nothing,
        dims=:,
        commutative::Bool=true,
        alg::Algorithm=Auto(),
        workspace=nothing,
    )

Apply `f` to each element of `src` and reduce the results with the binary operator `op`. With
`dims=:` (the default; `nothing` is accepted too) the result is a scalar, returned on the host.
With `dims` an integer or a collection of integers, the result is a new array on `src`'s backend
with size 1 along those dimensions; [`mapreducedim!`](@ref) reduces into an existing array
instead, and states the contract all reductions follow. In short:

- `op` must be associative, and commutative unless `commutative=false`, which combines the
  elements in their order.
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
    f::F, op::OP, src::MapReduceSource, srcs::AbstractArray...;
    backend::Union{Nothing, Backend}=nothing,
    init=NoInit(),
    neutral=nothing,
    acctype=nothing,
    dims=:,
    commutative::Bool=true,
    alg::Algorithm=Auto(),
    workspace=nothing,
) where {F, OP}
    f, src = fuse_sources(f, src, srcs)
    s = mapreduce_setup(f, op, src, backend, init, neutral, acctype, dims, alg)
    bufs = scratch_buffers(s.plan, workspace, src)
    return mapreduce_run(f, op, kernel_source(src, s.plan.backend), s, init, bufs; commutative)
end

function plan(
    ::typeof(mapreduce), f::F, op::OP, src::MapReduceSource, srcs::AbstractArray...;
    backend=nothing, init=NoInit(), neutral=nothing, acctype=nothing, dims=:,
    commutative::Bool=true, alg::Algorithm=Auto(),
) where {F, OP}
    f, src = fuse_sources(f, src, srcs)
    return mapreduce_setup(f, op, src, backend, init, neutral, acctype, dims, alg).plan
end

plan(::typeof(reduce), op::OP, src::AbstractArray; kwargs...) where {OP} =
    plan(mapreduce, identity, op, src; kwargs...)

# The source of a reduction, indexed from 1 (`one_based`): several arrays become one
# `Broadcasted` object
fuse_sources(f::F, src, ::Tuple{}) where {F} = (f, one_based(src))
function fuse_sources(f::F, src, srcs::Tuple) where {F}
    src isa AbstractArray ||
        throw(ArgumentError("a Broadcasted source cannot be combined with more arrays"))
    Base.all(a -> axes(a) == axes(src), srcs) ||
        throw(DimensionMismatch("all input arrays must have the same axes"))
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, one_based(src),
                                                               Base.map(one_based, srcs)...))
    return identity, bc
end
# Scalar *linear* indexing into a multidimensional `Broadcasted` object is only available on
# Julia 1.12, so earlier versions materialize it when the reduction runs (after the plan and the
# workspace checks, which see the original arrays); the plan treats it as a dense array.
kernel_source(src, backend) = src
kernel_source(src::Base.Broadcast.Broadcasted, backend) =
    VERSION < v"1.12-" ? materialize_source(src, backend) : src

# The seed of the partial results (`reduce_seed`), where an accumulator type exists
mapreduce_seed(op::OP, ::Type{A}, neutral) where {OP, A} =
    A === Union{} ? nothing : reduce_seed(op, A, neutral)

# Everything a reduction resolves before touching data: its backend, algorithm and scratch (the
# plan), accumulator type and partial-result seed, and for reductions along `dims` the result
function mapreduce_setup(f::F, op::OP, src, backend, init, neutral, acctype, dims,
                         alg) where {F, OP}
    backend = resolve_backend(backend, src)
    M = mapped_eltype(f, src)
    # The accumulator type, from `init`'s type
    A = resolve_acctype(op, init isa NoInit ? Union{} : typeof(init), M, acctype)

    if iswhole(dims)
        a = resolve_reduce(alg, backend, A, dims)
        seed = mapreduce_seed(op, A, neutral)
        # (the buffer's name does not depend on the length, so that results infer)
        sizes = if a isa BlockReduce && seed !== nothing
            n = mapreduce_1d_partials(length(src), a.block_size, a.items_per_thread,
                                      reduce_tuning(backend, A).target_blocks, a.switch_below)
            (; partials=scratch(typeof(seed), n))
        else
            (;)
        end
        return (; plan=Plan(backend, a, sizes), A=Val(A), seed)
    end

    rdims = reduced_dims(dims, ndims(src))
    # The result has the accumulator type; where there is none (`op` always throws), only an
    # empty reduction can succeed
    R = A !== Union{} ? A : !(init isa NoInit) ? typeof(init) : M === Union{} ? Nothing : M
    dst_sizes = ntuple(d -> d in rdims ? 1 : srcsize(src, d), ndims(src))
    a = resolve_reduce(alg, backend, A, rdims)
    seed = mapreduce_seed(op, A, neutral)
    sizes = mapreduce_nd_sizes(src, backend, a, A, seed, rdims, Base.prod(dst_sizes))
    return (; plan=Plan(backend, a, sizes), A=Val(A), seed, rdims, R=Val(R),
            dst_sizes)
end

# Run the reduction set up by `mapreduce_setup`, with the plan's scratch buffers `bufs`
function mapreduce_run(f::F, op::OP, src, s, init, bufs; commutative::Bool=true) where {F, OP}
    backend, a = s.plan.backend, s.plan.alg
    if !haskey(s, :rdims)
        return mapreduce_whole(f, op, src, backend, a, unval(s.A); init, seed=s.seed,
                               partials=get(bufs, :partials, nothing), commutative)
    end
    if init isa NoInit && Base.any(d -> srcsize(src, d) == 0, s.rdims) &&
       Base.prod(s.dst_sizes) > 0
        throw(ArgumentError(
            "reducing over an empty dimension is not allowed without `init`; pass `init`"))
    end
    dst = KernelAbstractions.allocate(backend, unval(s.R), s.dst_sizes)
    return mapreduce_nd!(f, op, dst, src, backend, a, unval(s.A);
                         init, seed=s.seed, rdims=s.rdims, bufs, commutative)
end

# A reduction nested in another operation, with the scratch buffers of its plan in the outer one
function mapreduce_nested(f::F, op::OP, src, bufs; backend, init=NoInit(), neutral=nothing,
                          dims=:, alg) where {F, OP}
    src = one_based(src)
    mapreduce_run(f, op, src,
                  mapreduce_setup(f, op, src, backend, init, neutral, nothing, dims, alg),
                  init, bufs)
end


"""
    mapreducedim!(
        f, op, R::AbstractArray, src;
        backend=get_backend(R),
        init=<none>,
        neutral=nothing,
        overwrite::Bool=false,
        acctype=nothing,
        commutative::Bool=true,
        alg::Algorithm=Auto(),
        workspace=nothing,
    ) -> R

Reduce `src` (an array or a `Broadcasted` object) into `R`: the dimensions along which `R` has
size 1 and `src` does not are reduced, and the others must match (`R` may leave off trailing
dimensions of size 1). `R` must not alias `src`.

This is the contract of every AcceleratedKernels reduction, modelled on CUB's rather than on
Base's (see [Differences from Base](@ref)):

- **Algebra.** `op` must be associative, and by default commutative: as CUB's reductions do, AK
  then combines elements in an order that depends on the algorithm, its settings and the shape,
  not in element order. With `commutative=false`, elements are combined in their order (each
  partial result reduces consecutive elements, and only neighbouring partial results are
  combined, the earlier on the left), so that operators such as matrix products are reduced
  correctly; the grouping is still unspecified, as in Base. The GPU kernels that keep the order
  can be slower, mostly for inputs that fit in the device's cache. Either way the order is fixed
  for a given setting and shape, so results are reproducible on one device.
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
    f::F, op::OP, R::AbstractArray, src::MapReduceSource;
    backend::Union{Nothing, Backend}=nothing,
    init=NoInit(),
    neutral=nothing,
    overwrite::Bool=false,
    acctype=nothing,
    commutative::Bool=true,
    alg::Algorithm=Auto(),
    workspace=nothing,
) where {F, OP}
    s = mapreducedim_setup(f, op, R, src, backend, init, neutral, overwrite, acctype, alg)
    mapreducedim_run!(f, op, R, src, s, workspace, commutative)
    return R
end

# (a function barrier: the type of `s` depends on the number of reduced dimensions)
function mapreducedim_run!(f::F, op::OP, R, src, s, workspace, commutative) where {F, OP}
    bufs = scratch_buffers(s.plan, workspace, R, src)
    backend = s.plan.backend
    mapreduce_nd!(f, op, s.dst, kernel_source(s.src, backend), backend, s.plan.alg, unval(s.A);
                  init=s.init, seed=s.seed, rdims=s.rdims, bufs, commutative)
end

plan(::typeof(mapreducedim!), f::F, op::OP, R::AbstractArray, src::MapReduceSource;
     backend=nothing, init=NoInit(), neutral=nothing, overwrite::Bool=false, acctype=nothing,
     commutative::Bool=true, alg::Algorithm=Auto()) where {F, OP} =
    mapreducedim_setup(f, op, R, src, backend, init, neutral, overwrite, acctype, alg).plan

function mapreducedim_setup(f::F, op::OP, R, src, backend, init, neutral, overwrite, acctype,
                            alg) where {F, OP}
    backend = resolve_backend(backend, R, src)
    nd = ndims(src)
    for d in 1:max(nd, ndims(R))
        sR, sA = size(R, d), srcsize(src, d)
        sR == 1 || sR == sA || throw(DimensionMismatch(
            "cannot reduce an array of size $(size(src)) into one of size $(size(R))"))
    end
    check_noalias(R, src)
    R, src = one_based(R), one_based(src)
    dst_sizes = ntuple(d -> size(R, d), nd)
    dst = size(R) == dst_sizes ? R : reshape(R, dst_sizes)
    rdims = Tuple(d for d in 1:nd if dst_sizes[d] == 1 && srcsize(src, d) != 1)
    return reducedim_setup(f, op, src, dst, rdims, backend, init, neutral, overwrite,
                           acctype, alg)
end
function reducedim_setup(f::F, op::OP, src, dst, rdims, backend, init, neutral,
                         overwrite, acctype, alg) where {F, OP}
    A = resolve_acctype(op, eltype(dst), mapped_eltype(f, src), acctype)
    a = resolve_reduce(alg, backend, A, rdims)
    seed = mapreduce_seed(op, A, neutral)
    sizes = mapreduce_nd_sizes(src, backend, a, A, seed, rdims, length(dst))
    init = init isa NoInit && !overwrite ? FoldInto() : init
    return (; plan=Plan(backend, a, sizes), src, dst, A=Val(A), seed, rdims, init)
end

function check_noalias(R, src::AbstractArray)
    Base.mightalias(R, src) &&
        throw(ArgumentError("the destination of a reduction must not alias its source"))
    nothing
end
check_noalias(R, src::Base.Broadcast.Broadcasted) = foreach(a -> check_noalias(R, a), src.args)
check_noalias(R, src::Base.Broadcast.Extruded) = check_noalias(R, src.x)
check_noalias(R, src) = nothing

# Reduce all of `src` to a host value, with accumulator type `A` and partial-result seed
# `seed`; `partials` is the plan's scratch
function mapreduce_whole(f::F, op::OP, src, backend, alg, ::Type{A}; init, seed,
                         partials, commutative) where {F, OP, A}
    if length(src) == 0
        init isa NoInit || return init
        throw(ArgumentError(
            "reducing over an empty collection is not allowed without `init`; pass `init`"))
    end
    # (a single element without `init` has nothing to combine, even when `op` always throws)
    if A === Union{} && length(src) == 1 && init isa NoInit
        return @allowscalar Base.mapreduce_first(f, op, src[first(eachindex(src))])
    end
    check_combinable(op, f, A)

    # The result, `op(init, partial)` or `partial`, has the accumulator type (asserted, as the
    # threaded reduction is not inferred)
    result = if alg isa BlockReduce
        mapreduce_1d_gpu(
            f, op, src, backend;
            init, seed,
            block_size=alg.block_size, items_per_thread=alg.items_per_thread,
            max_blocks=reduce_tuning(backend, A).target_blocks,
            partials, switch_below=alg.switch_below, commutative,
        )
    else
        # (behind a barrier: where `alg` is not inferred, e.g. a non-constant global, inference
        # would otherwise take this path for device arrays too, and through GPUArrays' `mapreduce`
        # of its host reductions recurse into `mapreduce` with ever more wrapped operators)
        Base.inferencebarrier(mapreduce_1d_cpu)(
            f, op, src, backend;
            init, seed,
            max_tasks=alg.max_tasks, min_elems=alg.min_elems,
        )
    end
    return convert(A, result)::A
end

include("shorthands.jl")
