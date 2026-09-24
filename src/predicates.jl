"""
    ConcurrentWrite(; block_size=nothing)

GPU `any`/`all` in which every thread whose element decides the result stores the same value to
one global flag. It is the fastest predicate algorithm, but some Intel GPUs hang when many
threads write one location, so it is rejected on backends that do not declare it safe (currently
oneAPI).
"""
Base.@kwdef struct ConcurrentWrite <: PredicateAlgorithm
    block_size::Union{Nothing, Int} = nothing
end

"""
    ViaReduce(; reduce::ReduceAlgorithm=BlockReduce())

`any`/`all` as a reduction with `|` or `&`, using the reduction algorithm `reduce`. It runs on
every backend that runs AcceleratedKernels' kernels, but reads every element.
"""
Base.@kwdef struct ViaReduce{R <: ReduceAlgorithm} <: PredicateAlgorithm
    reduce::R = BlockReduce()
end


"""
    PredicateTuning(; kwargs...)

Values that drive `Auto` selection and fill unset algorithm fields for `any` and `all` on one
device, as returned by [`predicate_tuning`](@ref):

- `prefer_concurrent_write`: `Auto` picks `ConcurrentWrite` where the backend supports it
  (`_supports_concurrent_write`), else `ViaReduce()`.
- `block_size`: for `ConcurrentWrite`.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.Partitioned`.

The defaults reproduce AK's historical settings. Internal: the fields may change in any release.
"""
Base.@kwdef struct PredicateTuning
    prefer_concurrent_write::Bool = true
    block_size::Int = 256
    threads_min_elems::Int = 1
end

"""
    predicate_tuning(backend, T) -> PredicateTuning

The `any`/`all` tuning for element type `T` (the input's) on `backend`'s current device; see
[`sort_tuning`](@ref) for the conventions.
"""
predicate_tuning(::Backend, ::Type) = PredicateTuning()

"""
    _resolve_predicate(alg, backend, T) -> Algorithm

Resolve `alg` for `any`/`all` over elements of type `T` on `backend`: `ConcurrentWrite`,
`ViaReduce` (with its reduction resolved) or `CPUThreads.Partitioned`, with every field set, or an
`ArgumentError`.
"""
function _resolve_predicate(alg::Algorithm, backend::Backend, ::Type{T}) where {T}
    _checkdomain(alg)
    t = predicate_tuning(backend, T)
    a = alg isa Auto ? _select_predicate(backend, t) : alg
    a = _fill(a, t, backend)
    _check_predicate(a, backend)
    return a
end

function _select_predicate(backend, t::PredicateTuning)
    _runs_threads(backend) && return CPUThreads.Partitioned()
    return _supports_concurrent_write(backend) && t.prefer_concurrent_write ?
        ConcurrentWrite() : ViaReduce()
end

function _checkdomain(a::ConcurrentWrite)
    _check_pow2(a, :block_size)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "ConcurrentWrite: `block_size` must be at most 1024, got $(a.block_size)"))
    nothing
end

_fill(a::ConcurrentWrite, t::PredicateTuning, backend) =
    ConcurrentWrite(something(a.block_size, t.block_size))
# The nested reduction reduces `Bool`s along the whole array
_fill(a::ViaReduce, t::PredicateTuning, backend) =
    ViaReduce(_resolve_reduce(a.reduce, backend, Bool, :))
_fill(a::CPUThreads.Partitioned, t::PredicateTuning, backend) = _fill_threads(a, t)
_fill(a::Algorithm, t::PredicateTuning, backend) =
    throw(ArgumentError("$(_algname(a)) is not an algorithm for `any` and `all`"))

function _check_predicate(a::ConcurrentWrite, backend)
    _checkdomain(a)
    _require_kernels(a, backend)
    _supports_concurrent_write(backend) || throw(ArgumentError(
        "ConcurrentWrite can hang devices of $(_backend_name(backend)); use `ViaReduce()`"))
    nothing
end
_check_predicate(a::ViaReduce, backend) = nothing       # its reduction was resolved by `_fill`
function _check_predicate(a::CPUThreads.Partitioned, backend)
    _checkdomain(a)
    _require_threads(a, backend)
end


@kernel cpu=false inbounds=true function _any_global!(out, pred, v_arg)
    v = _const_source(v_arg)
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    # Technically this is a race, but it doesn't matter as all threads would write the same value.
    # For example, CUDA F4.2 says "If a non-atomic instruction executed by a warp writes to the
    # same location in global memory for more than one of the threads of the warp, only one thread
    # performs a write and which thread does it is undefined."
    temp[0x1] = 0x0
    @synchronize()

    # The ndrange check already protects us from out of bounds access
    if pred(v[i])
        temp[0x1] = 0x1
    end

    @synchronize()
    if temp[0x1] != 0x0
        out[0x1] = 0x1
    end
end


"""
    any(pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto(), workspace=nothing)

Check if any element of `v` satisfies the predicate `pred` (i.e. some `pred(v[i]) == true`); `pred`
must return a `Bool`. Optimised differently to `mapreduce` due to shortcircuiting behaviour of
booleans.

**Other names**: not often implemented standalone on GPUs, typically included as part of a
reduction.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host, and on GPUs [`ConcurrentWrite`](@ref), or
[`ViaReduce`](@ref) on backends where concurrent writes are unsafe (oneAPI). `backend` is derived
from `v`; pass it for inputs that do not determine it, such as index ranges. `workspace` takes
the scratch memory of a [`workspace`](@ref) made for the same call.

On the host, multithreaded parallelisation is only worth it for large arrays, relatively expensive
predicates, and/or rare occurrence of true; use `CPUThreads.Partitioned(; max_tasks, min_elems)`
to only use parallelism when worth it in your application.

# Examples
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 100_000))
AK.any(x -> x < 1, v)
AK.any(x -> x < 1, v; alg=AK.ViaReduce(AK.BlockReduce(switch_below=100)))
```

Checking a more complex condition with unmaterialised index ranges:
```julia
function complex_any(x, y)
    AK.any(eachindex(x); backend=AK.get_backend(x)) do i
        x[i] < 0 && y[i] > 0
    end
end

complex_any(CuArray(rand(Float32, 100)), CuArray(rand(Float32, 100)))
```
"""
function any(pred, v::AbstractArray; backend::Union{Nothing, Backend}=nothing,
             alg::Algorithm=Auto(), workspace=nothing)
    p = _predicate_plan(pred, |, v, backend, alg)
    _any(pred, v, p.backend, p.alg, _buffers(p, workspace, v))
end

_plan(::typeof(any), pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto()) =
    _predicate_plan(pred, |, v, backend, alg)

# The predicate must return a `Bool`, as in Base. Where inference shows it cannot, the call fails
# before launching, with an `ArgumentError` instead of the kernel's `TypeError`.
# WORKAROUND(Intel NEO): not every backend reports an error thrown in a kernel (OpenCL on Intel's
# NEO driver does not: JuliaGPU/OpenCL.jl#501), so this check is the only error there. Once fixed,
# it could go, leaving Base's `TypeError` from the kernel.
function _check_bool_result(pred, v)
    isempty(v) && return nothing        # (Base never calls the predicate then)
    R = Base.promote_op(pred, eltype(v))
    R === Union{} || Bool <: R || throw(ArgumentError(
        "the predicate must return a `Bool`, but returns `$R` for elements of type `$(eltype(v))`"))
    nothing
end

# The plan of `any`/`all` (whose `ViaReduce` reduces with `op`): the flag of `ConcurrentWrite`,
# or the scratch of the reduction
function _predicate_plan(pred, op, v, backend, alg)
    _check_bool_result(pred, v)
    backend = _resolve_backend(backend, v)
    a = _resolve_predicate(alg, backend, eltype(v))
    sizes = if a isa ConcurrentWrite
        (; flag=_buffer(Int8, 1))
    elseif a isa ViaReduce
        (; reduce=_mapreduce_setup(_BoolValued(pred), op, v, backend, op === (|) ? false : true,
                                   nothing, nothing, :, a.reduce).plan.sizes)
    else
        (;)
    end
    return _Plan(backend, a, sizes)
end

function _any(pred, v, backend, alg::ConcurrentWrite, bufs)
    # `ndrange` must not be zero
    isempty(v) && return false
    out = bufs.flag
    fill!(out, Int8(0))
    _any_global!(backend, alg.block_size)(out, pred, v, ndrange=length(v))
    return @allowscalar(out[1]) != 0
end

_any(pred, v, backend, alg::ViaReduce, bufs) =
    _mapreduce_nested(_BoolValued(pred), |, v, bufs.reduce; backend, init=false, alg=alg.reduce)

function _any(pred, v, backend, alg::CPUThreads.Partitioned, bufs)
    overall = Ref(false)
    task_partition(length(v), alg.max_tasks, alg.min_elems) do irange
        for i in irange
            if pred(v[i])
                # Again, this is technically a thread race, but it doesn't matter as all threads
                # would write the same value; no data corruption can occur
                overall[] = true
                break
            end
        end
    end
    return overall[]
end


"""
    all(pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto(), workspace=nothing)

Check if all elements of `v` satisfy the predicate `pred` (i.e. all `pred(v[i]) == true`); `pred`
must return a `Bool`. The keywords are those of [`any`](@ref).

# Examples
```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Float32, 100_000))
AK.all(x -> x > 0, v)
```

Checking a more complex condition with unmaterialised index ranges:
```julia
function complex_all(x, y)
    AK.all(eachindex(x); backend=AK.get_backend(x)) do i
        x[i] > 0 && y[i] < 0
    end
end

complex_all(MtlArray(rand(Float32, 100)), MtlArray(rand(Float32, 100)))
```
"""
function all(pred, v::AbstractArray; backend::Union{Nothing, Backend}=nothing,
             alg::Algorithm=Auto(), workspace=nothing)
    p = _predicate_plan(pred, &, v, backend, alg)
    _all(pred, v, p.backend, p.alg, _buffers(p, workspace, v))
end

_plan(::typeof(all), pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto()) =
    _predicate_plan(pred, &, v, backend, alg)

_all(pred, v, backend, alg::ConcurrentWrite, bufs) = !_any(!pred, v, backend, alg, bufs)
_all(pred, v, backend, alg::ViaReduce, bufs) =
    _mapreduce_nested(_BoolValued(pred), &, v, bufs.reduce; backend, init=true, alg=alg.reduce)
_all(pred, v, backend, alg::CPUThreads.Partitioned, bufs) = !_any(!pred, v, backend, alg, bufs)
