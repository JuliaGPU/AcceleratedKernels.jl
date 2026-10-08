# Whole-array reduction on the GPU, in at most two launches (as CUB's `DeviceReduce`): a first
# pass of at most `max_blocks` blocks (the tuning's `target_blocks`), each reducing tiles of
# `block_size * items_per_thread` elements in turn (a grid-stride loop) to one partial result,
# and a second pass of one block that reduces those partials. Each pass stores its values with
# `_finish`, so the last one can write the result into a device array.

# NI and K are compile-time values so the local-memory size is static and the load loop unrolls.
# `neutral` seeds each thread's partial result (an empty `_Lane` when `op` has no known neutral
# element); `f === _Partials()` when `src` holds partial results of an earlier pass. Block
# `iblock` reduces the tiles `iblock`, `iblock + nblocks`, ... of `src`, and stores
# `_finish(op, init, dst, iblock + 1, partial)` in `dst[iblock + 1]`.
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_block!(
    @Const(src), dst, f, op, neutral, init, nblocks, ::Val{NI}, ::Val{K},
) where {NI, K}

    sdata = @localmem typeof(neutral) (NI,)
    f_lanes, op_lanes = _lanefuncs(f, op, neutral)
    len = length(src)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Consecutive threads load consecutive elements, while each thread advances by NI; full
    # tiles need no bounds checks, and only the last tile can be partial.
    acc = neutral
    tile = iblock * (NI * K)
    while tile + NI * K <= len
        for s in 0x0:(K - 0x1)
            acc = op_lanes(acc, f_lanes(src[tile + s * NI + ithread + 0x1]))
        end
        tile += nblocks * (NI * K)
    end
    if tile < len
        for s in 0x0:(K - 0x1)
            idx = tile + s * NI + ithread
            if idx < len
                acc = op_lanes(acc, f_lanes(src[idx + 0x1]))
            end
        end
    end
    sdata[ithread + 0x1] = acc

    @synchronize()

    @inline reduce_group!(@context, op_lanes, sdata, ithread)

    if ithread == 0x0
        dst[iblock + 0x1] = _finish(op, init, dst, iblock + 0x1, sdata[0x1])
    end
end

# The number of blocks of the first pass over `len` elements
_mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks) =
    min(cld(len, block_size * items_per_thread), max_blocks)

# Reduce the non-empty `src`; `neutral` is the partial-result seed of `_reduce_seed`, `init` is a
# value, `_NoInit()` or (with `dst`) `_Fold()`, and `partials` holds `_mapreduce_1d_partials`
# partial results (`nothing` if none are needed). Without `dst`, returns the result as a host
# value; with `dst`, stores it in `dst[1]` on the device.
function mapreduce_1d_gpu(
    f::F, op::OP, src::MapReduceSource, backend::Backend;
    init,
    neutral,
    block_size::Int,
    items_per_thread::Int,
    max_blocks::Int,
    partials::Union{Nothing, AbstractArray},
    switch_below::Int,
    dst::Union{Nothing, AbstractArray}=nothing,
) where {F, OP}
    @argcheck 1 <= block_size <= 1024
    @argcheck ispow2(block_size)
    @argcheck items_per_thread >= 1
    @argcheck max_blocks >= 1
    @argcheck switch_below >= 0

    f_host, op_host = _lanefuncs(f, op, neutral)
    len = length(src)

    # Degenerate cases, finished on the host
    if dst === nothing
        # `f` may index device arrays too
        len == 1 && return _finish(op, init, nothing, 0,
                                   @allowscalar(op_host(neutral, f_host(src[1]))))
        if len < switch_below
            h_src = _host_copy(src)
            return _finish(op, init, nothing, 0,
                           Base.mapreduce(f_host, op_host, h_src; init=neutral))
        end
    end

    kernel! = _mapreduce_block!(backend, block_size)
    NI, K = Val(block_size), Val(items_per_thread)
    # (later passes see the source as a view, so use one for the first pass too)
    src_view = _mapreduce_1d_src_view(src)
    blocks = _mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks)

    # The last pass stores the result: in `dst` with `init` applied, or the bare partial in the
    # slot after the first pass's partials, for the host
    last_dst, last_init = dst === nothing ?
        (@view(partials[blocks + 1:blocks + 1]), _NoFinish()) : (dst, init)
    if blocks == 1
        Base.inferencebarrier(kernel!)(src_view, last_dst, f, op, neutral, last_init, 1, NI, K;
                                       ndrange=(block_size,))
    else
        p = @view partials[1:blocks]
        Base.inferencebarrier(kernel!)(src_view, p, f, op, neutral, _NoFinish(), blocks, NI, K;
                                       ndrange=(block_size * blocks,))
        if dst === nothing && blocks < switch_below
            return _finish(op, init, nothing, 0, Base.reduce(op_host, Vector(p); init=neutral))
        end
        # (the second pass reads the partials in unrolled steps of up to 16 loads per thread: in
        # one step for the default settings)
        K2 = Val(clamp(cld(max_blocks, block_size), 1, 16))
        Base.inferencebarrier(kernel!)(p, last_dst, _Partials(), op, neutral, last_init, 1,
                                       NI, K2; ndrange=(block_size,))
    end
    dst === nothing || return dst
    return _finish(op, init, nothing, 0, @allowscalar(last_dst[1]))
end

_host_copy(src::AbstractArray) = Array(src)
# A `Broadcasted` source is evaluated on the host from host copies of its arrays, so that it needs
# no device memory
_host_copy(src::Base.Broadcast.Broadcasted) = Base.Broadcast.materialize(_on_host(src))
_on_host(bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.Broadcasted(bc.f, Base.map(_on_host, bc.args), bc.axes)
_on_host(x::Base.Broadcast.Extruded) = _on_host(x.x)
_on_host(x::AbstractArray) = Array(x)
_on_host(x::AbstractRange) = x
_on_host(x) = x

_mapreduce_1d_src_view(src::AbstractArray) = @view src[1:end]
_mapreduce_1d_src_view(src::Base.Broadcast.Broadcasted) = src

# The number of partial results of `mapreduce_1d_gpu` over `len` elements: one per block of the
# first pass, and for a host result (`device=false`) a slot for the last partial; 0 when it needs
# none
function _mapreduce_1d_partials(len, block_size, items_per_thread, max_blocks, switch_below;
                                device::Bool=false)
    if device
        len <= 1 && return 0
        blocks = _mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks)
        return blocks == 1 ? 0 : blocks
    end
    (len <= 1 || len < switch_below) && return 0
    return _mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks) + 1
end
