"""
    ConcurrentWrite(; block_size=nothing)

GPU `any`/`all` in which every thread whose element decides the result stores the same value to
one global flag. It is the fastest predicate algorithm, but some older Intel GPUs have been reported
to hang when many threads write one location, and it launches a thread per element, which Metal's
grid cannot hold for 2^32 elements or more, so `Auto` picks [`ViaReduce`](@ref) on oneAPI and
Metal.
"""
Base.@kwdef struct ConcurrentWrite <: PredicateAlgorithm
    block_size::Union{Nothing, Int} = nothing
end

"""
    ViaReduce(; reduce::ReduceAlgorithm=BlockReduce())

`any`/`all` as a reduction with `|` or `&`, using the reduction algorithm `reduce`. It runs on
every backend that runs AcceleratedKernels' kernels. Both algorithms read every element, but
this one takes the reduction's passes over partial results, where [`ConcurrentWrite`](@ref)
takes a single pass.
"""
Base.@kwdef struct ViaReduce{R <: ReduceAlgorithm} <: PredicateAlgorithm
    reduce::R = BlockReduce()
end


"""
    PredicateTuning(; kwargs...)

Values that drive `Auto` selection and fill unset algorithm fields for `any` and `all` on one
device, as returned by [`predicate_tuning`](@ref):

- `prefer_concurrent_write`: whether `Auto` picks `ConcurrentWrite`, rather than `ViaReduce()`.
- `block_size`: for `ConcurrentWrite`.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.Partitioned`.
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
    resolve_predicate(alg, backend, T) -> Algorithm

Resolve `alg` for `any`/`all` over elements of type `T` on `backend`: `ConcurrentWrite`,
`ViaReduce` (with its reduction resolved) or `CPUThreads.Partitioned`, with every field set, or an
`ArgumentError`.
"""
function resolve_predicate(alg::Algorithm, backend::Backend, ::Type{T}) where {T}
    t = predicate_tuning(backend, T)
    a = alg isa Auto ? select_predicate(backend, t) : alg
    a = fill_defaults(a, t, backend)
    check_predicate(a, backend)
    return a
end

function select_predicate(backend, t::PredicateTuning)
    runs_threads(backend) && return CPUThreads.Partitioned()
    return t.prefer_concurrent_write ? ConcurrentWrite() : ViaReduce()
end

function check_fields(a::ConcurrentWrite)
    check_block_size(a)
    nothing
end

fill_defaults(a::ConcurrentWrite, t::PredicateTuning, backend) =
    ConcurrentWrite(something(a.block_size, t.block_size))
# The nested reduction reduces `Bool`s along the whole array
fill_defaults(a::ViaReduce, t::PredicateTuning, backend) =
    ViaReduce(resolve_reduce(a.reduce, backend, Bool, :))
fill_defaults(a::CPUThreads.Partitioned, t::PredicateTuning, backend) = fill_threads(a, t)
fill_defaults(a::Algorithm, t::PredicateTuning, backend) =
    throw(ArgumentError("$(algname(a)) is not an algorithm for `any` and `all`"))

function check_predicate(a::ConcurrentWrite, backend)
    check_fields(a)
    require_kernels(a, backend)
end
check_predicate(a::ViaReduce, backend) = nothing       # its reduction was resolved by `fill_defaults`
function check_predicate(a::CPUThreads.Partitioned, backend)
    check_fields(a)
    require_threads(a, backend)
end


# `f`, required to return a `Bool` (Base's `_bool`), for `count` and the predicates
struct BoolResult{F} <: Function
    f::F
end
@inline (c::BoolResult)(x) = c.f(x)::Bool


@kernel cpu=false inbounds=true function any_kernel!(out, pred, @Const(v), first_index)
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    # Technically this is a race, but it doesn't matter as all threads would write the same value.
    # For example, CUDA F4.2 says "If a non-atomic instruction executed by a warp writes to the
    # same location in global memory for more than one of the threads of the warp, only one thread
    # performs a write and which thread does it is undefined."
    temp[0x1] = 0x0
    @synchronize()

    # The ndrange check already protects us from out of bounds access
    if pred(v[first_index + i - 0x1])
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

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host, and [`ConcurrentWrite`](@ref) on GPUs
([`ViaReduce`](@ref) on oneAPI and Metal). `backend` is derived from `v`; pass it for inputs
that do not determine it, such as index ranges. `workspace` takes the scratch memory of a
[`workspace`](@ref) made for the same call.

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
    p = predicate_setup(pred, |, false, v, backend, alg).plan
    any_run(pred, v, p.backend, p.alg, scratch_buffers(p, workspace, v))
end

plan(::typeof(any), pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto()) =
    predicate_setup(pred, |, false, v, backend, alg).plan

# The predicate must return a `Bool`, as in Base. Where inference shows it cannot, the call fails
# on the host, before launching, with an `ArgumentError` instead of an exception from the kernel
# (which GPU backends report without Base's `TypeError`).
function check_bool_result(pred, v)
    isempty(v) && return nothing        # (Base never calls the predicate then)
    R = Base.promote_op(pred, eltype(v))
    R === Union{} || Bool <: R || throw(ArgumentError(
        "the predicate must return a `Bool`, but returns `$R` for elements of type `$(eltype(v))`"))
    nothing
end

# The plan of `any`/`all` (whose `ViaReduce` reduces with `op` from `init`): the flag of
# `ConcurrentWrite`, or the scratch of the reduction
function predicate_setup(pred, op, init, v, backend, alg)
    check_bool_result(pred, v)
    backend = resolve_backend(backend, v)
    a = resolve_predicate(alg, backend, eltype(v))
    sizes = if a isa ConcurrentWrite
        (; flag=scratch(Int8, 1))
    elseif a isa ViaReduce
        (; reduce=mapreduce_setup(BoolResult(pred), op, v, backend, init, nothing, nothing, :,
                                  a.reduce).plan.sizes)
    else
        (;)
    end
    return (; plan=Plan(backend, a, sizes))
end

function any_run(pred, v, backend, alg::ConcurrentWrite, bufs)
    # `ndrange` must not be zero
    isempty(v) && return false
    out = bufs.flag
    fill!(out, Int8(0))
    any_kernel!(backend, alg.block_size)(out, pred, v, firstindex(v), ndrange=length(v))
    return @allowscalar(out[1]) != 0
end

any_run(pred, v, backend, alg::ViaReduce, bufs) =
    mapreduce_nested(BoolResult(pred), |, v, bufs.reduce; backend, init=false, alg=alg.reduce)

function any_run(pred, v, backend, alg::CPUThreads.Partitioned, bufs)
    found = Threads.Atomic{Bool}(false)
    first_index = firstindex(v)
    task_partition(length(v), alg.max_tasks, alg.min_elems) do irange
        for i in irange
            if pred(v[first_index + i - 1])
                found[] = true
                break
            end
        end
    end
    return found[]
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
    p = predicate_setup(pred, &, true, v, backend, alg).plan
    all_run(pred, v, p.backend, p.alg, scratch_buffers(p, workspace, v))
end

plan(::typeof(all), pred, v::AbstractArray; backend=nothing, alg::Algorithm=Auto()) =
    predicate_setup(pred, &, true, v, backend, alg).plan

all_run(pred, v, backend, alg::ConcurrentWrite, bufs) = !any_run(!pred, v, backend, alg, bufs)
all_run(pred, v, backend, alg::ViaReduce, bufs) =
    mapreduce_nested(BoolResult(pred), &, v, bufs.reduce; backend, init=true, alg=alg.reduce)
all_run(pred, v, backend, alg::CPUThreads.Partitioned, bufs) =
    !any_run(!pred, v, backend, alg, bufs)
