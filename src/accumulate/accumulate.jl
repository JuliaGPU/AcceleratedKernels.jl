include("tuning.jl")


# Implementations, then interfaces
include("accumulate_1d_cpu.jl")
include("accumulate_1d_gpu.jl")
include("accumulate_nd.jl")


"""
    accumulate!(op, v::AbstractArray; kwargs...) -> v
    accumulate!(op, dst::AbstractArray, src::AbstractArray; kwargs...) -> dst

    accumulate!(
        op, dst, src;
        backend=nothing,
        init=<none>,
        neutral=nothing,
        dims::Union{Nothing, Integer}=nothing,
        inclusive::Bool=true,
        acctype=nothing,
        alg::Algorithm=Auto(),
        workspace=nothing,
    )

Compute accumulated running totals along a sequence by applying a binary operator to all elements
up to the current one; often used in GPU programming as a first step in finding / extracting
subsets of data. The first form scans `v` in place; the second writes the scan of `src` to `dst`,
which must be `src` itself or not overlap it.

**Other names**: prefix sum, `thrust::scan`, cumulative sum; inclusive (or exclusive) if the first
element is included in the accumulation (or not).

With `dims=nothing` the array is scanned in linear order, whatever its number of dimensions; with
an integer `dims`, each slice along that dimension is scanned independently. A `dims` beyond the
array's dimensions makes every slice one element long.

The contract, which is AcceleratedKernels' own (see [Differences from Base](@ref)):

- **Algebra.** `op` must be associative, as elements are combined in parallel; it does not need
  to be commutative, as every combination keeps the elements in order (matrix products are
  fine). Non-associative operators such as `-` are not supported.
- **Inclusive scans** (the default): with running values `y`, `y[1] = Base.reduce_first(op,
  src[1])` without `init`, and `op(init, src[1])` with it; then `y[k] = op(y[k-1], src[k])`, and
  `dst[k] = y[k]`. `init` is applied once per slice.
- **Exclusive scans** (`inclusive=false`): `y[1]` is `init`, or the neutral element of `op`
  without it (an `ArgumentError` when none is known); then `y[k] = op(y[k-1], src[k-1])`.
- **Neutral element.** `neutral` (a two-sided identity of `op`) seeds partial results; it
  defaults to `GPUArraysCore.neutral_element(op, T)` where that is defined. For other operators,
  partial results start from their first element instead, so inclusive scans need no neutral
  element.
- **Running-value type.** The running values have one type, and are converted to `eltype(dst)`
  only when stored: `acctype` when it is given, else the type the fold of `op` settles on,
  starting from `eltype(dst)` (joined with `init`'s type) and the elements, as for
  [`mapreducedim!`](@ref) (e.g. `Int` for `add_sum` over `Int8`). So with the usual operators and
  no `acctype`, a scan does not run in a type narrower than its destination: `Float32`s scanned into a `Float64`
  array are summed in `Float64`. Where the running type differs from `eltype(dst)`, the scan runs in a scratch array
  of that type. An `acctype` that cannot hold the running values at all is an `ArgumentError`;
  whether their values fit is the caller's obligation. [`accumulate`](@ref) allocates the fold
  type from `init`'s type and the elements.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host, and on GPUs [`ScanPrefixes`](@ref) for
whole arrays and [`SliceScan`](@ref) along `dims`, with the device's settings. `backend` is
derived from `dst` and `src`.

On the host, accumulation is typically a memory-bound operation, so multithreaded accumulation
only becomes faster for more compute-heavy operations that hide memory latency, e.g. accumulating
tuples or structs, or expensive operators.

`workspace` takes the scratch memory of a [`workspace`](@ref) made for the same call, so that
the scan allocates none of its own.

# Examples
Example computing an inclusive prefix sum (the typical GPU "scan"):
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneAPI.ones(Int32, 100_000)
AK.accumulate!(+, v)                        # 1, 2, 3, ...
AK.accumulate!(+, v; inclusive=false)       # 0, 1, 3, ...: starts from the neutral element

# Choose the algorithm and its settings
AK.accumulate!(+, v; alg=AK.ScanPrefixes(block_size=512))
```
"""
function accumulate!(op, v::AbstractArray; backend::Union{Nothing, Backend}=nothing,
                     workspace=nothing, kwargs...)
    s = scan_setup(op, eltype(v), eltype(v), size(v), resolve_backend(backend, v);
                   kwargs...)
    w = one_based(v)
    scan_run!(op, w, w, s, scratch_buffers(s.plan, workspace, v))
    return v
end

function accumulate!(
    op, dst::AbstractArray, src::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    workspace=nothing,
    kwargs...
)
    check_scan_dst(dst, src)
    s = scan_setup(op, eltype(dst), eltype(src), size(dst),
                   resolve_backend(backend, dst, src); kwargs...)
    scan_run!(op, one_based(dst), one_based(src), s,
              scratch_buffers(s.plan, workspace, dst, src))
    return dst
end

plan(::typeof(accumulate!), op, v::AbstractArray; backend=nothing, kwargs...) =
    scan_setup(op, eltype(v), eltype(v), size(v), resolve_backend(backend, v);
               kwargs...).plan
function plan(::typeof(accumulate!), op, dst::AbstractArray, src::AbstractArray;
              backend=nothing, kwargs...)
    check_scan_dst(dst, src)
    scan_setup(op, eltype(dst), eltype(src), size(dst),
               resolve_backend(backend, dst, src); kwargs...).plan
end

function check_scan_dst(dst, src)
    dst === src && return nothing
    Base.mightalias(dst, src) && throw(ArgumentError(
        "the destination of a scan must be its source or not overlap it"))
    axes(dst) == axes(src) || throw(DimensionMismatch(
        "the destination of a scan must have the source's axes $(axes(src)), got $(axes(dst))"))
    nothing
end


# Everything a scan of elements of type `T` into an array of element type `D` and size `sz`
# resolves before touching data: its plan (backend, algorithm, scratch), running-value type and
# partial-result seed (see `reduce_seed`)
function scan_setup(
    op, ::Type{D}, ::Type{T}, sz::Dims, backend::Backend;
    init=NoInit(),
    neutral=nothing,
    dims::Union{Nothing, Integer}=nothing,
    inclusive::Bool=true,
    acctype=nothing,
    alg::Algorithm=Auto(),
) where {D, T}
    dims isa Integer && dims < 1 &&
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    n = Base.prod(sz; init=1)
    A = scan_acctype(op, D, T, init, inclusive, scan_calls_op(sz, dims, init, inclusive),
                     acctype)
    seed = reduce_seed(op, A, neutral)
    # The kernels' tiles hold partial results: `Optional`s, when `op` has no known neutral element
    a = resolve_scan(alg, backend, A, dims, typeof(seed))
    if !inclusive && init isa NoInit && seed isa Optional && !isset(seed)
        throw(ArgumentError(
            "an exclusive scan without `init` starts from the neutral element of `op`, " *
            "which is not known for $op; pass `init` or `neutral`"))
    end
    # A destination of another element type scans in a scratch array of the accumulator type
    work = D === A ? (;) : (; work=scratch(A, sz))
    prefixes = if a isa ScanPrefixes
        (; prefixes=scratch(typeof(seed), prefixes_length(n, a)))
    elseif a isa SliceScan
        # The chunk totals of slices split into chunks
        m = slice_scratch(sz, Int(dims), a, scan_tuning(backend, A), typeof(seed))
        m == 0 ? (;) : (; totals=scratch(typeof(seed), m))
    else
        (;)
    end
    return (; plan=Plan(backend, a, merge(work, prefixes)), A=Val(A), seed, init, dims,
            inclusive)
end

# Scan `src` into `v` (which may be `src`), as set up by `scan_setup`, with the plan's
# scratch buffers `bufs`
function scan_run!(op, v, src, s, bufs)
    backend, a = s.plan.backend, s.plan.alg
    init, seed, dims, inclusive = s.init, s.seed, s.dims, s.inclusive
    A = unval(s.A)

    # As Base, scan in the accumulator type and convert to `v`'s element type only when storing:
    # in `v` when it has that type, else in the scratch array. Elements enter as one-element
    # reductions (`Base.reduce_first`), and an inclusive scan applies `init` to the first element
    # of each slice, as `op(init, x)`. The kernels do both as they read `src`, for an `init` of
    # the accumulator type. The threaded algorithms, an `init` of another type, and slices of one
    # element (which no kernel scans) copy `src` to `w` first, apply `init` there, and scan `w`
    # in place.
    w = haskey(bufs, :work) ? bufs.work : v
    launch = launch_kwargs(a)
    singletons = dims !== nothing && dims > ndims(w)
    separate_init = inclusive && !(init isa NoInit) && !isempty(w) &&
                    (singletons || !(init isa A))
    in_place = a isa CPUThreads.Partitioned || separate_init || singletons
    if in_place && w !== src
        foreachindex_on(eachindex(w, src), backend; launch...) do i
            @inbounds w[i] = Base.reduce_first(op, src[i])
        end
    end
    if separate_init
        apply_init!(w, op, init, dims, backend, launch)
        init = NoInit()
    end
    if in_place || w === src
        scan_into!(op, w, w, identity, backend, a, bufs; init, seed, dims, inclusive)
    else
        scan_into!(op, w, src, ScanLoad{A}(op), backend, a, bufs; init, seed, dims, inclusive)
    end
    w === v || copyto_on!(v, w, backend)
    return v
end

# Scan into `w`, from the elements `load(src[i])` (`src` is `w`, with `load === identity`, for
# a scan in place; the threaded algorithms only scan in place)
function scan_into!(op::OP, w, src, load, backend, a, bufs; init, seed, dims,
                    inclusive) where {OP}
    if dims === nothing
        if a isa CPUThreads.Partitioned
            accumulate_1d_cpu!(op, w, backend, a; init, seed, inclusive)
        else
            accumulate_1d_gpu!(op, w, backend, a; src, load, init, seed, inclusive,
                               prefixes=get(bufs, :prefixes, nothing))
        end
    elseif dims > ndims(w)
        # Every slice has one element: inclusive scans applied `init` above, exclusive ones start
        # over
        inclusive || fill_on!(w, unwrap(scan_start(w, init, seed)), backend)
    else
        accumulate_nd!(op, w, backend, a; src, load, init, seed, dims=Int(dims), inclusive,
                       totals=get(bufs, :totals, nothing))
    end
    return w
end

# How a kernel reads an element of a scan's source: as a one-element reduction, of the
# accumulator type `A`, as the pass that copies the source to the destination stores it
struct ScanLoad{A, OP}
    op::OP
end
ScanLoad{A}(op::OP) where {A, OP} = ScanLoad{A, OP}(op)
@inline (e::ScanLoad{A})(x) where {A} = convert(A, Base.reduce_first(e.op, x))

# The value a scan starts from: `init` in the element type, else the seed
scan_start(v, init, seed) = lift(seed, convert(eltype(v), init))
scan_start(v, ::NoInit, seed) = seed

# The running-value type of a scan of elements of type `T` into an array of element type `D`
# (`Union{}` for none): `acctype`, else the accumulator type of a reduction from `D` joined with
# `init`'s type; an exclusive scan's `init` is a running value too. `combines` says whether `op`
# is called at all.
function scan_acctype(op, ::Type{D}, ::Type{T}, init, inclusive::Bool, combines::Bool,
                      acctype=nothing) where {D, T}
    acctype === nothing || return resolve_acctype(op, Union{}, T, acctype)
    S = init isa NoInit ? D : promote_type(D, typeof(init))
    A = accumulator_type(op, S, T)
    if !inclusive && !(init isa NoInit) && A !== Union{}
        A = promote_type(A, typeof(init))
    end
    if A === Union{}
        # `op` always throws for these types, which only a scan that never calls it gets past
        combines && check_combinable(op, identity, A)
        F = first_type(op, T)
        return F === Union{} ? T : F
    end
    return A
end

# Whether a scan of an array of size `sz` along `dims` calls `op`: some slice has two elements,
# or an inclusive scan applies `init` to a first element (along a `dims` beyond the array's
# dimensions, every element is one)
function scan_calls_op(sz, dims, init, inclusive)
    n = Base.prod(sz; init=1)
    (n == 0 || dims isa Integer && dims < 1) && return false
    len = dims === nothing ? n : dims <= length(sz) ? sz[dims] : 1
    return len >= 2 || (inclusive && !(init isa NoInit))
end

# `w[i] = op(init, w[i])` for the first element `i` of the array (`dims === nothing`) or of each
# slice along `dims` (every element, for a `dims` beyond the array's)
function apply_init!(w, op, init, dims, backend, launch)
    ax = axes(w)
    firsts = dims === nothing ? CartesianIndices(Base.map(a -> first(a):first(a), ax)) :
             dims > length(ax) ? CartesianIndices(ax) :
             CartesianIndices(Base.setindex(ax, first(ax[dims]):first(ax[dims]), dims))
    foreachindex_on(firsts, backend; launch...) do I
        @inbounds w[I] = op(init, w[I])
    end
    return w
end

# A scan nested in another operation, with the scratch buffers of its plan in the outer one
function accumulate_nested!(op, v, bufs; backend, kwargs...)
    s = scan_setup(op, eltype(v), eltype(v), size(v), backend; kwargs...)
    scan_run!(op, v, v, s, bufs)
end


"""
    accumulate(op, v::AbstractArray; init=<none>, kwargs...)

Out-of-place version of [`accumulate!`](@ref), with the same keywords. The result's element type
is `acctype` when it is given, else the type the fold of `op` settles on from `init`'s type (when
given) and the elements, e.g. an `Int` array for `accumulate(+, Int8[1, 2]; init=0)` and for
`accumulate(Base.add_sum, Int8[1, 2])`.
"""
function accumulate(op, v::AbstractArray; backend::Union{Nothing, Backend}=nothing,
                    init=NoInit(), dims=nothing, inclusive::Bool=true, acctype=nothing,
                    kwargs...)
    backend = resolve_backend(backend, v)
    D = accumulate_eltype(op, v, init, dims, inclusive, acctype)
    # (the result's axes start at 1, which `accumulate!` then requires of the source too)
    accumulate!(op, similar_on(backend, v, D), one_based(v); backend, init, dims, inclusive,
                acctype, kwargs...)
end

function plan(::typeof(accumulate), op, v::AbstractArray; backend=nothing, init=NoInit(),
              dims=nothing, inclusive::Bool=true, acctype=nothing, kwargs...)
    D = accumulate_eltype(op, v, init, dims, inclusive, acctype)
    scan_setup(op, D, eltype(v), size(v), resolve_backend(backend, v); init, dims,
               inclusive, acctype, kwargs...).plan
end

# The element type `accumulate` allocates: the running-value type from `init`'s type alone
accumulate_eltype(op, v, init, dims, inclusive, acctype) =
    scan_acctype(op, Union{}, eltype(v), init, inclusive,
                 scan_calls_op(size(v), dims, init, inclusive), acctype)


"""
    cumsum(src::AbstractArray; kwargs...)

Cumulative sum of elements of an array, as `Base.cumsum` (Base's `add_sum`, so small integers are
summed as `Int`), except that without `dims` a multidimensional array is summed in linear order.
The keywords are those of [`accumulate`](@ref).

# Examples
Simple cumulative sum of elements in a vector:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32(1):Int32(100), 100_000))
s = AK.cumsum(v)
```

Row-wise cumulative sum of a matrix:
```julia
m = ROCArray(rand(Int32(1):Int32(100), 10, 100_000))
s = AK.cumsum(m, dims=1)
```
"""
cumsum(src::AbstractArray; kwargs...) = accumulate(Base.add_sum, src; kwargs...)


"""
    cumprod(src::AbstractArray; kwargs...)

Cumulative product of elements of an array, as `Base.cumprod` (Base's `mul_prod`), except that
without `dims` a multidimensional array is multiplied in linear order. The keywords are those of
[`accumulate`](@ref).

# Examples
Simple cumulative product of elements in a vector:
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneArray(rand(Int32(1):Int32(100), 100_000))
p = AK.cumprod(v)
```

Row-wise cumulative product of a matrix:
```julia
m = oneArray(rand(Int32(1):Int32(100), 10, 100_000))
p = AK.cumprod(m, dims=1)
```
"""
cumprod(src::AbstractArray; kwargs...) = accumulate(Base.mul_prod, src; kwargs...)

plan(::typeof(cumsum), src::AbstractArray; kwargs...) = plan(accumulate, Base.add_sum, src; kwargs...)
plan(::typeof(cumprod), src::AbstractArray; kwargs...) = plan(accumulate, Base.mul_prod, src; kwargs...)
