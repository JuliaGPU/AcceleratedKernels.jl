# Selection and tuning of reduction algorithms; the steps are those of sorting (src/sort/tuning.jl).


"""
    BlockReduce(; block_size=nothing, items_per_thread=nothing, switch_below=nothing)

GPU tree reduction: each block of `block_size` threads (a power of two up to 1024) reduces
`block_size * items_per_thread` elements in local memory, repeatedly, until one value is left.
Reductions to a scalar (`dims=:`) finish on the host once fewer than `switch_below` values
remain; for them, `block_size * items_per_thread` must be between 2 and `typemax(Int32)`. Reductions along `dims` choose among several launch shapes by the sizes and strides of
the input and output; `items_per_thread` and `switch_below` do not apply to them, and setting
either is an `ArgumentError` there.

Like every GPU reduction in the Julia ecosystem, it requires an associative and commutative
operator: the combination order is fixed for a given setting and shape, but it is not the
element order.
"""
Base.@kwdef struct BlockReduce <: ReduceAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
    switch_below::Union{Nothing, Int} = nothing
end


"""
    ReduceTuning(; kwargs...)

Values that fill unset algorithm fields for reductions on one device, as returned by
[`reduce_tuning`](@ref).

- `block_size`, `items_per_thread`, `switch_below`: settings for unset `BlockReduce` fields.
- `target_blocks`: the number of blocks a reduction along `dims` aims to launch, to fill the
  device when there are few outputs; must be positive.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.Partitioned`.

The defaults reproduce AK's historical settings. Internal: the fields may change in any release.
"""
Base.@kwdef struct ReduceTuning
    block_size::Int = 256
    items_per_thread::Int = 2
    switch_below::Int = 0
    target_blocks::Int = 256
    threads_min_elems::Int = 1
end

"""
    reduce_tuning(backend, T) -> ReduceTuning

The reduction tuning for accumulator type `T` on `backend`'s current device; see
[`sort_tuning`](@ref) for the conventions.
"""
reduce_tuning(::Backend, ::Type) = ReduceTuning()


_whole(dims) = dims === nothing || dims isa Colon

"""
    _resolve_reduce(alg, backend, T, dims) -> Algorithm

Resolve `alg` for a reduction with accumulator type `T` along `dims` (`nothing` or `:` for a
whole-array reduction) on `backend`. Returns `BlockReduce` or `CPUThreads.Partitioned` with every
field set, or throws an `ArgumentError`.
"""
function _resolve_reduce(alg::Algorithm, backend::Backend, ::Type{T}, dims) where {T}
    _checkdomain(alg)
    if alg isa BlockReduce && !_whole(dims) &&
       (alg.items_per_thread !== nothing || alg.switch_below !== nothing)
        throw(ArgumentError("BlockReduce: `items_per_thread` and `switch_below` only apply to " *
                            "whole-array reductions (`dims=:`)"))
    end
    t = reduce_tuning(backend, T)
    a = alg isa Auto ? _select_reduce(backend) : alg
    a = _fill(a, t, T)
    _check_reduce(a, backend, T, dims)
    return a
end

_select_reduce(backend) = _runs_threads(backend) ? CPUThreads.Partitioned() : BlockReduce()

function _checkdomain(a::BlockReduce)
    _check_pow2(a, :block_size)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "BlockReduce: `block_size` must be at most 1024, got $(a.block_size)"))
    _check_positive(a, :items_per_thread)
    a.switch_below === nothing || a.switch_below >= 0 || throw(ArgumentError(
        "BlockReduce: `switch_below` must be non-negative, got $(a.switch_below)"))
    nothing
end

_fill(a::BlockReduce, t::ReduceTuning, T) =
    BlockReduce(something(a.block_size, t.block_size),
                something(a.items_per_thread, t.items_per_thread),
                something(a.switch_below, t.switch_below))
_fill(a::CPUThreads.Partitioned, t::ReduceTuning, T) = _fill_threads(a, t)
_fill(a::Algorithm, t::ReduceTuning, T) =
    throw(ArgumentError("$(_algname(a)) is not a reduction algorithm"))

function _check_reduce(a::BlockReduce, backend, ::Type{T}, dims) where {T}
    _checkdomain(a)
    _require_kernels(a, backend)
    # `Union{}`: no accumulator type, which only an empty reduction gets past
    T === Union{} || isbitstype(T) || throw(ArgumentError(
        "BlockReduce: the accumulator type $T is not a bits type; pass an `init` of the type " *
        "to accumulate in, or make `f` and `op` inferable"))
    if _whole(dims)
        # Each pass reduces tiles of `block_size * items_per_thread` elements to one value: a tile
        # of one element never shrinks the input, and the bound keeps the kernels' index
        # arithmetic far from overflow
        tile = widemul(a.block_size, a.items_per_thread)
        2 <= tile <= typemax(Int32) || throw(ArgumentError(
            "BlockReduce: `block_size * items_per_thread` must be between 2 and $(typemax(Int32)), " *
            "got $(a.block_size) * $(a.items_per_thread)"))
    end
    nothing
end

function _check_reduce(a::CPUThreads.Partitioned, backend, T, dims)
    _checkdomain(a)
    _require_threads(a, backend)
end
