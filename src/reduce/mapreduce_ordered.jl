# Reductions that combine elements in their order, for operators that need not be commutative
# (`commutative=false`). Every partial result is the reduction of a run of consecutive elements
# of one output's slice, and two partial results are only combined when they are neighbours,
# the earlier on the left.

# Block `iblock` reduces group `g` of output `iout` (`iblock = iout + g * output_size`): the
# outputs' slices are split into tiles of `N * K` elements, and each of the `groups` groups takes
# a contiguous range of them. In each tile, every thread reduces a run of `K` consecutive
# elements, and the block combines the runs in order (with the scans' block scan, which uses
# sub-groups where `subgroups` says so) and appends the tile's total to the group's. Indices and
# `f`, `neutral` and `init` are those of the other reduction kernels (see `_mapreduce_nd_by_block!`
# and `_mapreduce_block!`); with several groups, `dst` holds `output_size × groups` partial
# results, for a second pass of this kernel with `f === _Partials()`. With a vector width `W`, the
# runs of full tiles are read with `K ÷ W` aligned vector loads (see `_ordered_vector_width`).
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_ordered!(
    src, dst,
    f, op, neutral, init,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size, groups,
    ::Val{K}, ::Val{W}, subgroups::Val,
) where {K, W}
    @uniform N = @groupsize()[1]
    totals = @localmem typeof(neutral) (N,)
    f_lanes, op_lanes = _lanefuncs(f, op, neutral)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    # (unsigned division, see `_mapreduce_nd_by_thread_tiled_strided!`)
    iout = Int(unsigned(iblock) % unsigned(output_size))
    g    = Int(unsigned(iblock) ÷ unsigned(output_size))
    input_base = base_offset + _outer_decode(iout, outer_strides, outer_sizes)

    # This group's tiles, as evenly divided as they can be
    tile_size = N * K
    ntiles = cld(reduce_size, tile_size)
    tile = fld(g * ntiles, groups)
    stop = fld((g + 1) * ntiles, groups)

    total = neutral
    while tile < stop
        run = tile * tile_size + ithread * K
        acc = neutral
        if W > 0 && (tile + 1) * tile_size <= reduce_size
            for k in 0x0:W:(K - 0x1)
                v = _vloada(Val(W), pointer(src, input_base + run + k + 0x1))
                acc = _acc_lanes(f_lanes, op_lanes, acc, v)
            end
        elseif (tile + 1) * tile_size <= reduce_size
            for k in 0x0:(K - 0x1)
                off = _reduce_offset(run + k, reduce_strides, reduce_sizes)
                acc = op_lanes(acc, f_lanes(src[input_base + off + 0x1]))
            end
        else
            for k in 0x0:(K - 0x1)
                j = run + k
                if j < reduce_size
                    off = _reduce_offset(j, reduce_strides, reduce_sizes)
                    acc = op_lanes(acc, f_lanes(src[input_base + off + 0x1]))
                end
            end
        end
        totals[ithread + 0x1] = acc
        _, tile_total = block_exclusive_scan!(@context, op_lanes, totals, neutral, N, ithread,
                                              subgroups)
        total = op_lanes(total, tile_total)
        tile += 1
    end

    if ithread == 0x0
        i = iout + g * output_size + 0x1
        dst[i] = _finish(op, init, dst, i, total)
    end
end

# One pass of `_mapreduce_ordered!`, of `groups` blocks per output; `dense_buffer` is whether
# `src` is a dense buffer, which it can read with vector loads
function _mapreduce_ordered_pass!(
    backend, block_size, items_per_thread, src, dst, f::F, op::OP, neutral, init,
    base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
    output_size, reduce_size, groups; dense_buffer::Bool=false,
) where {F, OP}
    W = dense_buffer ? _ordered_vector_width(src, base_offset, outer_strides, reduce_strides,
                                             reduce_size, block_size, items_per_thread) : 0
    kernel! = _mapreduce_ordered!(backend, block_size)
    Base.inferencebarrier(kernel!)(
        src, dst, f, op, neutral, init,
        base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
        output_size, reduce_size, groups, Val(items_per_thread), Val(W),
        _scan_subgroups(backend, typeof(neutral), block_size),
        ndrange=(block_size * output_size * groups,),
    )
    return nothing
end

# The vector width of the loads of `_mapreduce_ordered!`: that of the by-block kernel's (see
# `_contiguous_vector_width`), where it divides the runs of `items_per_thread` elements; else 0
function _ordered_vector_width(buffer, base_offset, outer_strides, reduce_strides, reduce_size,
                               block_size, items_per_thread)
    W = _contiguous_vector_width(buffer, base_offset, outer_strides, reduce_strides, reduce_size,
                                 block_size)
    return W > 0 && items_per_thread % W == 0 ? W : 0
end

# Reduce the slices of `output_size` outputs into `dst` in element order: with one group per
# output in one pass, or with `groups` in two, the second reducing the groups' partial results in
# `partials` (of `output_size × groups` elements, the groups of each output `output_size` apart).
function _launch_mapreduce_ordered!(
    backend, block_size, items_per_thread, src, dst, f::F, op::OP, neutral, init,
    base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
    output_size, reduce_size, groups, partials; dense_buffer::Bool,
) where {F, OP}
    if groups == 1
        _mapreduce_ordered_pass!(backend, block_size, items_per_thread, src, dst, f, op,
                                 neutral, init, base_offset, outer_strides, outer_sizes,
                                 reduce_strides, reduce_sizes, output_size, reduce_size, 1;
                                 dense_buffer)
        return nothing
    end
    _mapreduce_ordered_pass!(backend, block_size, items_per_thread, src, partials, f, op,
                             neutral, _NoFinish(), base_offset, outer_strides, outer_sizes,
                             reduce_strides, reduce_sizes, output_size, reduce_size, groups;
                             dense_buffer)
    _mapreduce_ordered_pass!(backend, _pass2_block_size(groups), 1, partials, dst, _Partials(),
                             op, neutral, init, 0, (1,), (output_size,), (output_size,),
                             (groups,), output_size, groups, 1)
    return nothing
end
