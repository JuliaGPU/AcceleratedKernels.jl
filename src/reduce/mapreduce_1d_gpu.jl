# Whole-array reduction on the GPU, in at most two launches (as CUB's `DeviceReduce`): a first
# pass of at most `max_blocks` blocks (the tuning's `target_blocks`), each reducing tiles of
# `block_size * items_per_thread` elements in turn (a grid-stride loop) to one partial result,
# and a second pass of one block that reduces those partials. Each pass stores its values with
# `finish`, so the last one can write the result into a device array.

# NI and K are compile-time values so the local-memory size is static and the load loop unrolls.
# `seed` seeds each thread's partial result (an empty `Optional` when `op` has no known neutral
# element); `f === PartialInputs()` when `src` holds partial results of an earlier pass. Block
# `iblock` reduces the tiles `iblock`, `iblock + nblocks`, ... of `src`, and stores
# `finish(op, init, dst, iblock + 1, partial)` in `dst[iblock + 1]`.
@kernel inbounds=true cpu=false unsafe_indices=true function mapreduce_1d_kernel!(
    @Const(src), dst, f, op, seed, init, nblocks, ::Val{NI}, ::Val{K},
) where {NI, K}

    sdata = @localmem typeof(seed) (NI,)
    mapf, accop = seeded_ops(f, op, seed)
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
    acc = seed
    tile = iblock * (NI * K)
    while tile + NI * K <= len
        for s in 0x0:(K - 0x1)
            acc = accop(acc, mapf(src[tile + s * NI + ithread + 0x1]))
        end
        tile += nblocks * (NI * K)
    end
    if tile < len
        for s in 0x0:(K - 0x1)
            idx = tile + s * NI + ithread
            if idx < len
                acc = accop(acc, mapf(src[idx + 0x1]))
            end
        end
    end
    sdata[ithread + 0x1] = acc

    @synchronize()

    @inline reduce_group!(@context, accop, sdata, ithread)

    if ithread == 0x0
        dst[iblock + 0x1] = finish(op, init, dst, iblock + 0x1, sdata[0x1])
    end
end

# The number of blocks of the first pass over `len` elements
mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks) =
    min(cld(len, block_size * items_per_thread), max_blocks)

# Reduce the non-empty `src`; `seed` is the partial-result seed of `reduce_seed`, `init` is a
# value, `NoInit()` or (with `dst`) `FoldInto()`, and `partials` holds the partial results of
# `mapreduce_1d_partials` (with `dst`) or `mapreduce_1d_host_partials` (`nothing` if none are
# needed). Without `dst`, returns the result as a host value; with `dst`, stores it in `dst[1]`
# on the device.
function mapreduce_1d_gpu(
    f::F, op::OP, src::MapReduceSource, backend::Backend;
    init,
    seed,
    block_size::Int,
    items_per_thread::Int,
    max_blocks::Int,
    partials::Union{Nothing, AbstractArray},
    switch_below::Int,
    commutative::Bool=true,
    dst::Union{Nothing, AbstractArray}=nothing,
) where {F, OP}
    f_host, op_host = seeded_ops(f, op, seed)
    len = length(src)

    # Degenerate cases, finished on the host
    if dst === nothing
        # `f` may index device arrays too
        len == 1 && return finish(op, init, nothing, 0,
                                  @allowscalar(op_host(seed, f_host(src[1]))))
        if len < switch_below
            h_src = host_copy(src)
            return finish(op, init, nothing, 0,
                          Base.mapreduce(f_host, op_host, h_src; init=seed))
        end
    end

    # (later passes see the source as a view, so use one for the first pass too)
    src_view = mapreduce_1d_view(src)
    blocks = mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks)

    # The last pass stores the result: in `dst` with `init` applied, or the bare partial in the
    # slot after the first pass's partials, for the host
    last_dst, last_init = dst === nothing ?
        (@view(partials[blocks + 1:blocks + 1]), Unfinished()) : (dst, init)
    buffer = commutative ? nothing : mapreduce_1d_buffer(src)
    if blocks == 1
        mapreduce_1d_pass!(commutative, backend, block_size, items_per_thread, src_view,
                           last_dst, f, op, seed, last_init, 1; buffer)
    else
        p = @view partials[1:blocks]
        mapreduce_1d_pass!(commutative, backend, block_size, items_per_thread, src_view, p,
                           f, op, seed, Unfinished(), blocks; buffer)
        if dst === nothing && blocks < switch_below
            return finish(op, init, nothing, 0, Base.reduce(op_host, Vector(p); init=seed))
        end
        # (the second pass reads the partials in unrolled steps of up to 16 loads per thread: in
        # one step for the default settings)
        mapreduce_1d_pass!(commutative, backend, block_size,
                           clamp(cld(max_blocks, block_size), 1, 16), p, last_dst,
                           PartialInputs(), op, seed, last_init, 1)
    end
    dst === nothing || return dst
    return finish(op, init, nothing, 0, @allowscalar(last_dst[1]))
end

# A pass of `blocks` blocks over the linearly indexed `src`, each storing a partial result in
# `dst`: with the strided kernel, or in element order, reading `buffer` (see
# `mapreduce_1d_buffer`) where one is given
function mapreduce_1d_pass!(commutative::Bool, backend, block_size, items_per_thread, src, dst,
                            f::F, op::OP, seed, init, blocks; buffer=nothing) where {F, OP}
    if commutative
        kernel! = mapreduce_1d_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(src, dst, f, op, seed, init, blocks, Val(block_size),
                                       Val(items_per_thread); ndrange=(block_size * blocks,))
    else
        len = length(src)
        src, base_offset = something(buffer, (src, 0))
        ordered_pass!(backend, block_size, items_per_thread, src, dst, f, op,
                      seed, init, base_offset, (), (), (1,), (len,), 1, len, blocks;
                      dense_buffer=buffer !== nothing)
    end
end

# A contiguous source as `(buffer, base_offset)`, its elements' dense buffer and their offset in
# it, which the ordered kernel reads with vector loads; `nothing` for other sources
function mapreduce_1d_buffer(src)
    src isa AbstractArray || return nothing
    layout = strided_layout(src)
    layout !== nothing && layout[3] == dense_strides(size(src)) || return nothing
    return layout[1], layout[2]
end

host_copy(src::AbstractArray) = Array(src)
# A `Broadcasted` source is evaluated on the host from host copies of its arrays, so that it needs
# no device memory
host_copy(src::Base.Broadcast.Broadcasted) = Base.Broadcast.materialize(on_host(src))
on_host(bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.Broadcasted(bc.f, Base.map(on_host, bc.args), bc.axes)
on_host(x::Base.Broadcast.Extruded) = on_host(x.x)
on_host(x::AbstractArray) = Array(x)
on_host(x::AbstractRange) = x
on_host(x) = x

# (a view of all elements, indexed from 1 also when the source's indices are not)
mapreduce_1d_view(src::AbstractArray) = @view src[begin:end]
mapreduce_1d_view(src::Base.Broadcast.Broadcasted) = src

# The number of partial results `mapreduce_1d_gpu` stores over `len` elements into `dst`: one
# per block of the first pass, or none when one block reduces them all
function mapreduce_1d_partials(len, block_size, items_per_thread, max_blocks)
    len <= 1 && return 0
    blocks = mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks)
    return blocks == 1 ? 0 : blocks
end
# ... and for a host result: one per block of the first pass and a slot for the last pass's
# partial result, or none when the host reduces them (fewer than `switch_below` elements)
function mapreduce_1d_host_partials(len, block_size, items_per_thread, max_blocks, switch_below)
    (len <= 1 || len < switch_below) && return 0
    return mapreduce_1d_blocks(len, block_size, items_per_thread, max_blocks) + 1
end
