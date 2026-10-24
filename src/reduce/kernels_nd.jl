# The kernels of a reduction along `dims`, and their index helpers.

# Index helpers. Both decode a linear index into a byte-offset by walking compile-time-sized
# segment tuples. The single-segment case (the common one after canonicalization) needs no
# division; multi-segment falls back to a per-element decode (rare: non-adjacent dim sets).

@inline function outer_offset(tid, outer_strides, outer_sizes)
    isempty(outer_sizes) && return 0
    length(outer_sizes) == 1 && return tid * outer_strides[1]
    base = 0
    tmp  = tid
    @inbounds for i in 1:length(outer_sizes)
        q     = tmp ÷ outer_sizes[i]
        r     = tmp - q * outer_sizes[i]
        base += r * outer_strides[i]
        tmp   = q
    end
    base
end

@inline function reduce_offset(j, reduce_strides, reduce_sizes)
    isempty(reduce_sizes) && return 0
    # Single segment: j < reduce_size, so the decode is just j * stride — no division.
    length(reduce_sizes) == 1 && return j * reduce_strides[1]
    off = 0
    tmp = j
    @inbounds for i in 1:length(reduce_sizes) - 1
        sz = reduce_sizes[i]
        if ispow2(sz)
            shift = trailing_zeros(sz)
            r   = tmp & (sz - 1)
            tmp = tmp >> shift
        else
            q   = tmp ÷ sz
            r   = tmp - q * sz
            tmp = q
        end
        off += r * reduce_strides[i]
    end
    off += tmp * reduce_strides[end]
    off
end

# GPU kernel: one thread per output element, reducing sequentially over the reduced extents
# using Cartesian indexing; the by_thread kernel of sources without a dense buffer. Mirrors the
# CPU `mapreduce_nd_threads!` path.
@kernel inbounds=true cpu=false unsafe_indices=true function by_thread_cartesian_kernel!(
    src, dst,
    f, op, seed, init,
    Rother, Rreduce, output_size,
)
    mapf, accop = seeded_ops(f, op, seed)
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < output_size
        Iother = Rother[tid + 0x1]
        res = seed
        for Ireduce in Rreduce
            J = max(Iother, Ireduce)
            res = accop(res, mapf(src[J]))
        end
        dst[Iother] = finish(op, init, dst, Iother, res)
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function tiled_strided_kernel!(
    @Const(src), dst,
    f, op, seed, init,
    base_offset, reduce_stride,
    output_size, reduce_size,
    ::Val{rows},
) where {rows}
    @uniform N = @groupsize()[1]
    @uniform reduce_threads = N ÷ rows
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Int(...) keeps the index decode signed; a bare `unsigned ÷ unsigned` result feeds
    # the signed index math and adds an Int-conversion throw guard NVPTX keeps (CUDA ~20%).
    row  = Int(unsigned(ithread) % unsigned(rows))
    lane = Int(unsigned(ithread) ÷ unsigned(rows))
    iout = iblock * rows + row

    acc = seed
    if iout < output_size
        # Contiguous outputs (outer_strides == (1,)), so the source base is just iout;
        # base_offset (0 for a dense source) is folded in once, outside the reduce loop.
        sbase = base_offset + iout
        j = lane
        while j < reduce_size
            acc = accop(acc, mapf(src[sbase + j * reduce_stride + 0x1]))
            j += reduce_threads
        end
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    step = reduce_threads ÷ 2
    while step >= 1
        if lane < step
            dst_lane = row + lane * rows
            src_lane = row + (lane + step) * rows
            sdata[dst_lane + 0x1] = accop(sdata[dst_lane + 0x1], sdata[src_lane + 0x1])
        end
        @synchronize()
        step ÷= 2
    end

    if lane == 0x0 && iout < output_size
        dst[iout + 0x1] = finish(op, init, dst, iout + 0x1, sdata[row + 0x1])
    end
end

# GPU kernel: segments — each output reduces `reduce_size` consecutive elements (from the offset
# of its outer index), with `lanes` threads that read them in an interleaved order; a block of
# `N` threads covers `N ÷ lanes` outputs, which are combined in local memory with log2(lanes)
# steps.
@kernel inbounds=true cpu=false unsafe_indices=true function by_lanes_kernel!(
    @Const(src), dst,
    f, op, seed, init,
    base_offset, outer_strides, outer_sizes,
    output_size, reduce_size, ::Val{lanes},
) where {lanes}
    @uniform N = @groupsize()[1]
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # (unsigned division by a constant, see `tiled_strided_kernel!`)
    lane = Int(unsigned(ithread) % unsigned(lanes))
    row  = Int(unsigned(ithread) ÷ unsigned(lanes))
    iout = iblock * (N ÷ lanes) + row

    acc = seed
    if iout < output_size
        input_base = base_offset + outer_offset(iout, outer_strides, outer_sizes)
        j = lane
        while j < reduce_size
            acc = accop(acc, mapf(src[input_base + j + 0x1]))
            j += lanes
        end
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    step = lanes ÷ 2
    while step >= 1
        if lane < step
            sdata[ithread + 0x1] = accop(sdata[ithread + 0x1], sdata[ithread + step + 0x1])
        end
        @synchronize()
        step ÷= 2
    end

    if lane == 0x0 && iout < output_size
        dst[iout + 0x1] = finish(op, init, dst, iout + 0x1, sdata[ithread + 0x1])
    end
end

# GPU kernel: columns — a long strided reduction into contiguous outputs, e.g. the row sums of
# a wide column-major matrix (`dims=2`). Each block covers `cols` consecutive outputs of a run of
# `run` (the outputs at an offset of the other kept dimensions, `run_strides` and `run_sizes`),
# with `N ÷ cols` lanes each, so that a warp's loads of one step are consecutive elements, and the
# reduction of each output is split across `groups` blocks. With one group the block stores the
# outputs; with more, `dst` holds `output_size × groups` partial results for a second pass (and
# `init` is `Unfinished()`). The lanes and groups take their elements in turn, or `ordered`, each a
# contiguous range of them, combined in order; either way a warp's loads of one step are
# consecutive elements.
@kernel inbounds=true cpu=false unsafe_indices=true function columns_kernel!(
    @Const(src), dst,
    f, op, seed, init,
    base_offset, reduce_stride,
    output_size, run, runs, run_strides, run_sizes, reduce_size, groups,
    ::Val{cols}, ::Val{ordered},
) where {cols, ordered}
    @uniform N = @groupsize()[1]
    @uniform lanes = N ÷ cols
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    run_tiles = cld(run, cols)
    tiles = run_tiles * runs

    # (unsigned divisions by constants and by the tile counts; see the tiled kernel. The block
    # index is divided in 32 bits: the columns kernel takes fewer than 64 Ki outputs (see
    # `strided_shape`), in tiles of at least 8, split into groups only to approach
    # `target_blocks` blocks. 64-bit divisions take it 7% longer on an RTX 6000 Ada)
    col  = Int(unsigned(ithread) % unsigned(cols))
    lane = Int(unsigned(ithread) ÷ unsigned(cols))
    tile  = Int((iblock % UInt32) % (tiles % UInt32))
    group = Int((iblock % UInt32) ÷ (tiles % UInt32))
    irun = isempty(run_sizes) ? 0 : Int((tile % UInt32) ÷ (run_tiles % UInt32))
    icol = (tile - irun * run_tiles) * cols + col
    valid = icol < run
    iout = irun * run + icol
    sbase = base_offset + outer_offset(irun, run_strides, run_sizes) + icol

    acc = seed
    if ordered && valid
        # (the group's range of the slice, and the lane's of that, as evenly divided as they can be)
        lo = fld(group * reduce_size, groups)
        len = fld((group + 1) * reduce_size, groups) - lo
        j = lo + fld(lane * len, lanes)
        stop = lo + fld((lane + 1) * len, lanes)
        while j + (COLUMNS_ITEMS_PER_THREAD - 1) < stop
            for k in 0:(COLUMNS_ITEMS_PER_THREAD - 1)
                acc = accop(acc, mapf(src[sbase + (j + k) * reduce_stride + 0x1]))
            end
            j += COLUMNS_ITEMS_PER_THREAD
        end
        while j < stop
            acc = accop(acc, mapf(src[sbase + j * reduce_stride + 0x1]))
            j += 1
        end
    elseif valid
        step = groups * lanes
        j = group * lanes + lane
        # (several loads in flight per thread: one at a time leaves the kernel at half the
        # bandwidth)
        while j + (COLUMNS_ITEMS_PER_THREAD - 1) * step < reduce_size
            for k in 0:(COLUMNS_ITEMS_PER_THREAD - 1)
                acc = accop(acc, mapf(src[sbase + (j + k * step) * reduce_stride + 0x1]))
            end
            j += COLUMNS_ITEMS_PER_THREAD * step
        end
        while j < reduce_size
            acc = accop(acc, mapf(src[sbase + j * reduce_stride + 0x1]))
            j += step
        end
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    if ordered
        # (each step combines neighbouring ranges, the earlier on the left)
        s = 1
        while s < lanes
            if lane % (2 * s) == 0
                sdata[col + lane * cols + 0x1] =
                    accop(sdata[col + lane * cols + 0x1], sdata[col + (lane + s) * cols + 0x1])
            end
            @synchronize()
            s *= 2
        end
    else
        s = lanes ÷ 2
        while s >= 1
            if lane < s
                sdata[col + lane * cols + 0x1] =
                    accop(sdata[col + lane * cols + 0x1], sdata[col + (lane + s) * cols + 0x1])
            end
            @synchronize()
            s ÷= 2
        end
    end

    if lane == 0x0 && valid
        i = iout + group * output_size + 0x1
        dst[i] = finish(op, init, dst, i, sdata[col + 0x1])
    end
end

# GPU kernel: by_thread — one thread per output element, reducing sequentially.
# Used when there are more output elements than elements in the reduced dimension(s),
# e.g. reduce(+, rand(3, 1000), dims=1) — only 3 elements to reduce per output.

@kernel inbounds=true cpu=false unsafe_indices=true function by_thread_kernel!(
    @Const(src), dst,
    f, op, seed, init,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size,
)
    mapf, accop = seeded_ops(f, op, seed)
    # NOTE: index calculations use zero-indexing (fewer ops, matches the CUDA / ROCm / oneAPI /
    # Metal code this is transpiled to), converting to one-indexing only at memory accesses.
    # `base_offset` (the source's element offset within its dense buffer; 0 for a dense
    # source) is folded into the per-output base, so it costs at most one add per thread.
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < output_size
        input_base = base_offset + outer_offset(tid, outer_strides, outer_sizes)

        res = seed
        for j in 0x0:reduce_size - 0x1
            off = reduce_offset(j, reduce_strides, reduce_sizes)
            res = accop(res, mapf(src[input_base + off + 0x1]))
        end

        dst[tid + 0x1] = finish(op, init, dst, tid + 0x1, res)
    end
end

# GPU kernel: by_block — each block reduces one output element, then grid-strides to
# the next output (iout += num_blocks) until all outputs are covered. When
# num_blocks >= output_size, each block handles at most one output (the original,
# non-striding behavior) at zero extra cost — the while loop runs once.
#
# Used for FEW_OUTPUTS <= dst_size < reduce_size (grid-stride, num_blocks ==
# min(dst_size, target_blocks) < dst_size in general), and also for the
# reduce_groups==1 fallback inside the multigroup branch (num_blocks == dst_size,
# so the loop runs exactly once per block — identical to the pre-grid-stride kernel).

@kernel inbounds=true cpu=false unsafe_indices=true function by_block_kernel!(
    @Const(src), dst,
    f, op, seed, init,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size, num_blocks,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    iout = iblock
    while true
        input_base = base_offset + outer_offset(iout, outer_strides, outer_sizes)

        acc = seed
        j   = ithread
        while j < reduce_size
            off = reduce_offset(j, reduce_strides, reduce_sizes)
            acc = accop(acc, mapf(src[input_base + off + 0x1]))
            j  += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, accop, sdata, ithread)

        if ithread == 0x0
            dst[iout + 0x1] = finish(op, init, dst, iout + 0x1, sdata[0x1])
        end

        next_iout = iout + num_blocks
        next_iout >= output_size && break

        @synchronize()
        iout = next_iout
    end
end

# Stride-1 by-block reduction using one aligned 128-bit load per thread iteration.
# The host launcher verifies the pointer, row alignment, and UInt32 index bound.
@kernel inbounds=true cpu=false unsafe_indices=true function by_block_vector_kernel!(
    src, dst,
    f, op, seed, init,
    base_offset,
    outer_strides, outer_sizes,
    output_size, reduce_size, num_blocks,
    valW,
)
    @uniform N = @groupsize()[1]
    @uniform T = eltype(src)
    @uniform W = unval(valW)
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    iout = iblock
    while true
        input_base = base_offset + outer_offset(iout, outer_strides, outer_sizes)

        acc     = seed
        rs      = UInt32(reduce_size)
        jw      = UInt32(ithread) * UInt32(W)
        stridew = UInt32(N) * UInt32(W)
        while jw + UInt32(W - 1) < rs
            v = vload(Val(W), pointer(src, input_base + jw + 0x1))
            acc = fold_vector(mapf, accop, acc, v)
            jw += stridew
        end
        while jw < rs
            acc = accop(acc, mapf(src[input_base + jw + 0x1]))
            jw += UInt32(1)
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, accop, sdata, ithread)

        if ithread == 0x0
            dst[iout + 0x1] = finish(op, init, dst, iout + 0x1, sdata[0x1])
        end

        next_iout = iout + num_blocks
        next_iout >= output_size && break

        @synchronize()
        iout = next_iout
    end
end

# GPU kernel: multi-group first pass — several blocks per output element. Block `iblock`
# handles output `iout` and group `igroup`, reducing its interleaved slice of the reduction
# into partial[iout, igroup]. The second pass combines the groups.

@kernel inbounds=true cpu=false unsafe_indices=true function multigroup_kernel!(
    @Const(src), partial,
    f, op, seed,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    ::Val{output_size}, reduce_size, reduce_groups,
) where {output_size}
    @uniform N = @groupsize()[1]
    sdata = @localmem typeof(seed) (N,)
    mapf, accop = seeded_ops(f, op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Int(...) keeps the index decode signed (see tiled kernel): avoids a CUDA throw guard.
    iout   = Int(unsigned(iblock) % unsigned(output_size))
    igroup = Int(unsigned(iblock) ÷ unsigned(output_size))

    input_base = base_offset + outer_offset(iout, outer_strides, outer_sizes)

    acc = seed
    j   = ithread + igroup * N
    while j < reduce_size
        off = reduce_offset(j, reduce_strides, reduce_sizes)
        acc = accop(acc, mapf(src[input_base + off + 0x1]))
        j  += N * reduce_groups
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    @inline reduce_group!(@context, accop, sdata, ithread)

    if ithread == 0x0
        partial[iout + igroup * output_size + 0x1] = sdata[0x1]
    end
end

# GPU kernel: multi-group second pass — one block per output reduces over the `reduce_groups`
# partials and applies `init` (see `finish`). Launched with a block size sized to `reduce_groups`
# (see combine_block_size), avoiding an oversubscribed tree-reduce when reduce_groups
# is much smaller than 256.

@kernel inbounds=true cpu=false unsafe_indices=true function combine_groups_kernel!(
    @Const(partial), dst,
    op, seed, init,
    output_size, reduce_groups,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem typeof(seed) (N,)
    _, accop = seeded_ops(PartialInputs(), op, seed)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        acc = seed
        g = ithread
        while g < reduce_groups
            acc = accop(acc, partial[iblock + g * output_size + 0x1])
            g += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, accop, sdata, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = finish(op, init, dst, iblock + 0x1, sdata[0x1])
        end
    end
end
