# NI and K are compile-time values so the local-memory size is static and the load loop unrolls.
# `neutral` seeds each thread's partial result (an empty `_Lane` when `op` has no known neutral
# element); `f === _Partials()` when `src` holds partial results of an earlier pass.
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_block!(
    src_arg, dst, f, op, neutral, ::Val{NI}, ::Val{K},
) where {NI, K}
    src = _const_source(src_arg)

    sdata = @localmem typeof(neutral) (NI,)
    f, op = _lanefuncs(f, op, neutral)
    len = length(src)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Consecutive threads load consecutive elements, while each thread advances by NI.
    acc = neutral
    for s in 0x0:(K - 0x1)
        idx = iblock * (NI * K) + s * NI + ithread
        if idx < len
            acc = op(acc, f(src[idx + 0x1]))
        end
    end
    sdata[ithread + 0x1] = acc

    @synchronize()

    @inline reduce_group!(@context, op, sdata, ithread)

    if ithread == 0x0
        dst[iblock + 0x1] = sdata[0x1]
    end
end


# Reduce the non-empty `src` to a host value; `neutral` is the partial-result seed of
# `_reduce_seed`, and `init` is a value or `_NoInit()`.
function mapreduce_1d_gpu(
    f, op, src::MapReduceSource, backend::Backend;
    init,
    neutral,
    block_size::Int,
    items_per_thread::Int,
    temp::Union{Nothing, AbstractArray},
    switch_below::Int,
)
    @argcheck 1 <= block_size <= 1024
    @argcheck ispow2(block_size)
    @argcheck items_per_thread >= 1
    @argcheck switch_below >= 0

    P = typeof(neutral)
    f_host, op_host = _lanefuncs(f, op, neutral)

    # Degenerate cases
    len = length(src)
    # `f` may index device arrays too
    len == 1 && return _finish(op, init, nothing, 0,
                               @allowscalar(op_host(neutral, f_host(src[1]))))
    if len < switch_below
        h_src = _host_copy(src)
        return _finish(op, init, nothing, 0, Base.mapreduce(f_host, op_host, h_src; init=neutral))
    end

    # Each block handles `items_per_thread * block_size` elements.
    num_per_block = items_per_thread * block_size
    blocks = (len + num_per_block - 1) ÷ num_per_block

    if !isnothing(temp)
        neutral isa _Lane && throw(ArgumentError(
            "`temp` needs a neutral element of the operator; pass `neutral`"))
        @argcheck get_backend(temp) === backend
        eltype(temp) === P || throw(ArgumentError(
            "`temp` must have the reduction's accumulator element type $P, got $(eltype(temp))"))
        @argcheck length(temp) >= blocks * 2
        dst = temp
    else
        dst = KernelAbstractions.allocate(backend, P, blocks * 2)
    end

    # Later the kernel will be compiled for views anyways, so use same types for arrays.
    src_view = _mapreduce_1d_src_view(src)
    dst_view = @view dst[1:blocks]

    kernel! = _mapreduce_block!(backend, block_size)
    kernel!(src_view, dst_view, f, op, neutral, Val(block_size), Val(items_per_thread);
            ndrange=(block_size * blocks,))

    # As long as we still have blocks to process, swap between the src and dst pointers at
    # the beginning of the first and second halves of dst
    len = blocks
    if len < switch_below
        h_src = Vector(@view(dst[1:len]))
        return _finish(op, init, nothing, 0, Base.reduce(op_host, h_src; init=neutral))
    end

    # Now all src elements have been passed through f; just do final reduction, no map needed
    p1 = @view dst[1:len]
    p2 = @view dst[blocks + 1:end]

    while len > 1
        blocks = (len + num_per_block - 1) ÷ num_per_block

        # Each block produces one reduced value
        kernel!(p1, p2, _Partials(), op, neutral, Val(block_size), Val(items_per_thread);
                ndrange=(block_size * blocks,))
        len = blocks

        if len < switch_below
            h_src = Vector(@view(p2[1:len]))
            return _finish(op, init, nothing, 0, Base.reduce(op_host, h_src; init=neutral))
        end

        p1, p2 = p2, p1
        p1 = @view p1[1:len]
    end

    # The GPU kernel reduced all elements to one; apply init
    return _finish(op, init, nothing, 0, @allowscalar(p1[1]))
end

_host_copy(src::AbstractArray) = Array(src)
_host_copy(src::Base.Broadcast.Broadcasted) = Array(Base.Broadcast.materialize(src))

_mapreduce_1d_src_view(src::AbstractArray) = @view src[1:end]
_mapreduce_1d_src_view(src::Base.Broadcast.Broadcasted) = src
