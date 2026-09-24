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
whole arrays and [`SliceScan`](@ref) along `dims`, with the device's settings.
[`DecoupledLookback`](@ref) is available on backends that support it. `backend` is derived from
`dst` and `src`.

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
    s = _accumulate_setup(op, eltype(v), eltype(v), size(v), _resolve_backend(backend, v);
                          kwargs...)
    _accumulate_run!(op, v, v, s, _buffers(s.plan, workspace, v))
end

function accumulate!(
    op, dst::AbstractArray, src::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    workspace=nothing,
    kwargs...
)
    _check_scan_destination(dst, src)
    s = _accumulate_setup(op, eltype(dst), eltype(src), size(dst),
                          _resolve_backend(backend, dst, src); kwargs...)
    _accumulate_run!(op, dst, src, s, _buffers(s.plan, workspace, dst, src))
end

_plan(::typeof(accumulate!), op, v::AbstractArray; backend=nothing, kwargs...) =
    _accumulate_setup(op, eltype(v), eltype(v), size(v), _resolve_backend(backend, v);
                      kwargs...).plan
function _plan(::typeof(accumulate!), op, dst::AbstractArray, src::AbstractArray;
               backend=nothing, kwargs...)
    _check_scan_destination(dst, src)
    _accumulate_setup(op, eltype(dst), eltype(src), size(dst),
                      _resolve_backend(backend, dst, src); kwargs...).plan
end

function _check_scan_destination(dst, src)
    dst === src && return nothing
    Base.mightalias(dst, src) && throw(ArgumentError(
        "the destination of a scan must be its source or not overlap it"))
    axes(dst) == axes(src) || throw(DimensionMismatch(
        "the destination of a scan must have the source's axes $(axes(src)), got $(axes(dst))"))
    nothing
end


# Everything a scan of elements of type `T` into an array of element type `D` and size `sz`
# resolves before touching data: its plan (backend, algorithm, scratch), running-value type and
# partial-result seed (see `_reduce_seed`)
function _accumulate_setup(
    op, ::Type{D}, ::Type{T}, sz::Dims, backend::Backend;
    init=_NoInit(),
    neutral=nothing,
    dims::Union{Nothing, Integer}=nothing,
    inclusive::Bool=true,
    acctype=nothing,
    alg::Algorithm=Auto(),
) where {D, T}
    dims isa Integer && dims < 1 &&
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    n = Base.prod(sz; init=1)
    A = _scan_acctype(op, D, T, init, inclusive, _scan_combines(sz, dims, init, inclusive),
                      acctype)
    seed = _reduce_seed(op, A, neutral)
    # The kernels' tiles hold partial results: lanes, when `op` has no known neutral element
    a = _resolve_scan(alg, backend, A, dims, typeof(seed))
    if !inclusive && init isa _NoInit && seed isa _Lane && !_valid(seed)
        throw(ArgumentError(
            "an exclusive scan without `init` starts from the neutral element of `op`, " *
            "which is not known for $op; pass `init` or `neutral`"))
    end
    # A destination of another element type scans in a scratch array of the accumulator type
    work = D === A ? (;) : (; work=_buffer(A, sz))
    prefixes = if a isa Union{ScanPrefixes, DecoupledLookback}
        num_blocks = cld(n, a.block_size * a.items_per_thread)
        p = _buffer(typeof(seed), num_blocks)
        a isa DecoupledLookback ? (; prefixes=p, flags=_buffer(UInt8, num_blocks)) :
                                  (; prefixes=p)
    else
        (;)
    end
    return (; plan=_Plan(backend, a, merge(work, prefixes)), A=Val(A), seed, init, dims,
            inclusive)
end

# Scan `src` into `v` (which may be `src`), as set up by `_accumulate_setup`, with the plan's
# scratch buffers `bufs`
function _accumulate_run!(op, v, src, s, bufs)
    backend, a = s.plan.backend, s.plan.alg
    init, neutral, dims, inclusive = s.init, s.seed, s.dims, s.inclusive
    A = _unval(s.A)

    # As Base, scan in the accumulator type and convert to `v`'s element type only when storing:
    # in `v` when it has that type, else in the scratch array. Elements enter as one-element
    # reductions (`Base.reduce_first`), and an inclusive scan applies `init` to the first element
    # of each slice, as `op(init, x)`.
    w = haskey(bufs, :work) ? bufs.work : v
    launch = _scan_launch(a)
    if w !== src
        _foreachindex(eachindex(w, src), backend; launch...) do i
            @inbounds w[i] = Base.reduce_first(op, src[i])
        end
    end
    # (`init` of the accumulator type seeds the kernels directly instead, except where every
    # slice has one element and no kernel runs)
    if inclusive && !(init isa _NoInit) && !isempty(w) &&
       (!(init isa A) || dims !== nothing && dims > ndims(w))
        _scan_apply_init!(w, op, init, dims, backend, launch)
        init = _NoInit()
    end

    if dims === nothing
        if a isa CPUThreads.Partitioned
            accumulate_1d_cpu!(op, w, backend, a; init, neutral, inclusive)
        else
            accumulate_1d_gpu!(op, w, backend, a; init, neutral, inclusive,
                               prefixes=get(bufs, :prefixes, nothing),
                               flags=get(bufs, :flags, nothing))
        end
    elseif dims > ndims(w)
        # Every slice has one element: inclusive scans applied `init` above, exclusive ones start
        # over
        inclusive || fill!(w, _unlane(_scan_first_seed(w, init, neutral)))
    else
        accumulate_nd!(op, w, backend, a; init, neutral, dims=Int(dims), inclusive)
    end
    w === v || copyto!(v, w)
    return v
end

# The running-value type of a scan of elements of type `T` into an array of element type `D`
# (`Union{}` for none): `acctype`, else the accumulator type of a reduction from `D` joined with
# `init`'s type; an exclusive scan's `init` is a running value too. `combines` says whether `op`
# is called at all.
function _scan_acctype(op, ::Type{D}, ::Type{T}, init, inclusive::Bool, combines::Bool,
                       acctype=nothing) where {D, T}
    acctype === nothing || return _acctype(op, Union{}, T, acctype)
    S = init isa _NoInit ? D : promote_type(D, typeof(init))
    A = _reduce_acctype(op, S, T)
    if !inclusive && !(init isa _NoInit) && A !== Union{}
        A = promote_type(A, typeof(init))
    end
    if A === Union{}
        # `op` always throws for these types, which only a scan that never calls it gets past
        combines && _check_acctype(op, identity, A)
        F = _first_type(op, T)
        return F === Union{} ? T : F
    end
    return A
end

# Whether a scan of an array of size `sz` along `dims` calls `op`: some slice has two elements,
# or an inclusive scan applies `init` to a first element (along a `dims` beyond the array's
# dimensions, every element is one)
function _scan_combines(sz, dims, init, inclusive)
    n = Base.prod(sz; init=1)
    (n == 0 || dims isa Integer && dims < 1) && return false
    len = dims === nothing ? n : dims <= length(sz) ? sz[dims] : 1
    return len >= 2 || (inclusive && !(init isa _NoInit))
end

_scan_launch(a::Union{ScanPrefixes, DecoupledLookback, SliceScan}) = (; block_size=a.block_size)
_scan_launch(a::CPUThreads.Partitioned) = (; max_tasks=a.max_tasks, min_elems=a.min_elems)

# `w[i] = op(init, w[i])` for the first element `i` of the array (`dims === nothing`) or of each
# slice along `dims` (every element, for a `dims` beyond the array's)
function _scan_apply_init!(w, op, init, dims, backend, launch)
    ax = axes(w)
    firsts = dims === nothing ? CartesianIndices(Base.map(a -> first(a):first(a), ax)) :
             dims > length(ax) ? CartesianIndices(ax) :
             CartesianIndices(Base.setindex(ax, first(ax[dims]):first(ax[dims]), dims))
    _foreachindex(firsts, backend; launch...) do I
        @inbounds w[I] = op(init, w[I])
    end
    return w
end

# A scan nested in another operation, with the scratch buffers of its plan in the outer one
function _accumulate_nested!(op, v, bufs; backend, kwargs...)
    s = _accumulate_setup(op, eltype(v), eltype(v), size(v), backend; kwargs...)
    _accumulate_run!(op, v, v, s, bufs)
end


"""
    accumulate(op, v::AbstractArray; init=<none>, kwargs...)

Out-of-place version of [`accumulate!`](@ref), with the same keywords. The result's element type
is `acctype` when it is given, else the type the fold of `op` settles on from `init`'s type (when
given) and the elements, e.g. an `Int` array for `accumulate(+, Int8[1, 2]; init=0)` and for
`accumulate(Base.add_sum, Int8[1, 2])`.
"""
function accumulate(op, v::AbstractArray; backend::Union{Nothing, Backend}=nothing,
                    init=_NoInit(), dims=nothing, inclusive::Bool=true, acctype=nothing,
                    kwargs...)
    backend = _resolve_backend(backend, v)
    D = _accumulate_eltype(op, v, init, dims, inclusive, acctype)
    accumulate!(op, _similar(backend, v, D), v; backend, init, dims, inclusive, acctype,
                kwargs...)
end

function _plan(::typeof(accumulate), op, v::AbstractArray; backend=nothing, init=_NoInit(),
               dims=nothing, inclusive::Bool=true, acctype=nothing, kwargs...)
    D = _accumulate_eltype(op, v, init, dims, inclusive, acctype)
    _accumulate_setup(op, D, eltype(v), size(v), _resolve_backend(backend, v); init, dims,
                      inclusive, acctype, kwargs...).plan
end

# The element type `accumulate` allocates: the running-value type from `init`'s type alone
_accumulate_eltype(op, v, init, dims, inclusive, acctype) =
    _scan_acctype(op, Union{}, eltype(v), init, inclusive,
                  _scan_combines(size(v), dims, init, inclusive), acctype)
