# NI = threads/block, K = items/thread — both compile-time (`Val`), so the K-loop unrolls and
# `@localmem` sizing is static. K == 1 loads one element/thread; K == 2 reproduces the historical
# two-elements-per-thread load; K > 2 amortises launch + indexing overhead on capable GPUs.
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_block!(
    @Const(src), dst, f, op, neutral, ::Val{NI}, ::Val{K},
) where {NI, K}

    sdata = @localmem eltype(dst) (NI,)
    len = length(src)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Block owns a K*NI-wide window; fold this thread's K block-strided items with `op`.
    # `op(neutral, x) == x`, so starting from `neutral` reproduces the old "first element as-is" load.
    tile = translate_base(Tile((i = NI * K,)), (i = iblock * NI * K,))
    acc = neutral
    for k in 0:K - 1
        idx = translate_offset(tile, (i = ithread + k * NI,)).index.i   # 0-based
        if idx < len
            acc = op(acc, f(src[idx + 0x1]))
        end
    end
    sdata[ithread + 0x1] = acc

    @synchronize()

    @inline reduce_group!(@context, op, sdata, NI, ithread)

    if ithread == 0x0
        dst[iblock + 0x1] = sdata[0x1]
    end
end


function mapreduce_1d_gpu(
    f, op, src::MapReduceSource, backend::Backend;
    init,
    neutral,

    # CPU settings - ignored here
    max_tasks::Int,
    min_elems::Int,

    # GPU settings
    block_size::Int,
    items_per_thread::Int,
    temp::Union{Nothing, AbstractArray},
    switch_below::Int,
)
    @argcheck 1 <= block_size <= 1024
    @argcheck ispow2(block_size)
    @argcheck items_per_thread >= 1
    @argcheck switch_below >= 0

    # Degenerate cases
    len = length(src)
    len == 0 && return init
    len == 1 && return op(init, @allowscalar f(src[1]))
    if len < switch_below
        h_src = Vector(src)
        return Base.mapreduce(f, op, h_src; init)
    end

    # Each thread handles `items_per_thread` elements (block covers items_per_thread * block_size)
    num_per_block = items_per_thread * block_size
    blocks = (len + num_per_block - 1) ÷ num_per_block

    # Compile-time specialization parameters threaded into the kernel as `Val`s
    valNI = Val(block_size)
    valK  = Val(items_per_thread)

    if !isnothing(temp)
        @argcheck get_backend(temp) === backend
        @argcheck eltype(temp) === typeof(init)
        @argcheck length(temp) >= blocks * 2
        dst = temp
    else
        # Figure out type for destination
        dst_type = typeof(init)
        dst = KernelAbstractions.allocate(backend, dst_type, blocks * 2)
    end

    # Later the kernel will be compiled for views anyways, so use same types for arrays.
    src_view = _mapreduce_1d_src_view(src)
    dst_view = @view dst[1:blocks]

    kernel! = _mapreduce_block!(backend, block_size)
    kernel!(src_view, dst_view, f, op, neutral, valNI, valK, ndrange=(block_size * blocks,))

    # As long as we still have blocks to process, swap between the src and dst pointers at
    # the beginning of the first and second halves of dst
    len = blocks
    if len < switch_below
        h_src = Vector(@view(dst[1:len]))
        return Base.reduce(op, h_src; init)
    end

    # Now all src elements have been passed through f; just do final reduction, no map needed
    p1 = @view dst[1:len]
    p2 = @view dst[blocks + 1:end]

    while len > 1
        blocks = (len + num_per_block - 1) ÷ num_per_block

        # Each block produces one reduced value
        kernel!(p1, p2, identity, op, neutral, valNI, valK, ndrange=(block_size * blocks,))
        len = blocks

        if len < switch_below
            h_src = Vector(@view(p2[1:len]))
            return Base.reduce(op, h_src; init)
        end

        p1, p2 = p2, p1
        p1 = @view p1[1:len]
    end

    # The GPU kernel reduced all elements to one, but without the init value
    return op(init, @allowscalar(p1[1]))
end

_mapreduce_1d_src_view(src::AbstractArray) = @view src[1:end]
_mapreduce_1d_src_view(src::Base.Broadcast.Broadcasted) = src
