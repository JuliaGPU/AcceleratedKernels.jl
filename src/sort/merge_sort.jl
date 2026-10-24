@kernel inbounds=true cpu=false unsafe_indices=true function merge_tile_kernel!(
    v, comp, layout, blocks_per_slice,
)

    @uniform N = @groupsize()[1]
    s_buf = @localmem eltype(v) (N * 0x2,)

    T = eltype(v)
    I = typeof(N)

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block sorts a tile of one slice
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    elems = slice(v, layout, islice)
    len = layout.len

    i = ithread + iblock * N * 0x2
    i < len && (s_buf[ithread + 0x1] = elems[i + 0x1])

    i = ithread + N + iblock * N * 0x2
    i < len && (s_buf[ithread + N + 0x1] = elems[i + 0x1])

    @synchronize()

    run_len = typeof(ithread)(1)
    pair_len = typeof(ithread)(2)

    while run_len <= N
        ipair = ithread ÷ run_len

        local v1::T
        local v2::T
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = ipair * pair_len + run_len + iblock * N * 0x2
        if i < len
            tid = ipair * pair_len + ithread % run_len
            v1 = s_buf[tid + 0x1]

            i = (ipair + 0x1) * pair_len + iblock * N * 0x2
            n = i < len ? run_len : len - iblock * N * 0x2 - ipair * pair_len - run_len
            lo = ipair * pair_len + run_len
            hi = lo + n
            pos1 = ithread % run_len + lower_bound0(s_buf, v1, lo, hi, comp) - lo
        end

        tid = ipair * pair_len + run_len + ithread % run_len
        i = tid + iblock * N * 0x2
        if i < len
            v2 = s_buf[tid + 0x1]
            lo = ipair * pair_len
            hi = lo + run_len
            pos2 = ithread % run_len + upper_bound0(s_buf, v2, lo, hi, comp) - lo
        end

        @synchronize()

        pos1 != typemax(I) && (s_buf[ipair * pair_len + pos1 + 0x1] = v1)
        pos2 != typemax(I) && (s_buf[ipair * pair_len + pos2 + 0x1] = v2)

        @synchronize()

        run_len = run_len << 0x1
        pair_len = pair_len << 0x1
    end

    i = ithread + iblock * N * 0x2
    i < len && (elems[i + 0x1] = s_buf[ithread + 0x1])

    i = ithread + N + iblock * N * 0x2
    i < len && (elems[i + 0x1] = s_buf[ithread + N + 0x1])
end


@kernel inbounds=true cpu=false unsafe_indices=true function merge_pass_kernel!(
    @Const(v_in), v_out, comp, run_len, layout, blocks_per_slice,
)
    N = @groupsize()[1]

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Merges never cross slice boundaries
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    slice_in = slice(v_in, layout, islice)
    slice_out = slice(v_out, layout, islice)
    len = layout.len

    idx = ithread + iblock * N
    pair_len = run_len * 0x2
    ipair = idx ÷ run_len

    # Left half
    pos_in = ipair * pair_len + idx % run_len
    lo = ipair * pair_len + run_len

    if lo >= len
        # Incomplete left half, nothing to swap on the right, simply copy elements to be sorted
        # in next iteration
        pos_in < len && (slice_out[pos_in + 0x1] = slice_in[pos_in + 0x1])
    else

        hi = (ipair + 0x1) * pair_len
        hi > len && (hi = len)

        pos_out = pos_in + lower_bound0(slice_in, slice_in[pos_in + 0x1], lo, hi, comp) - lo
        slice_out[pos_out + 0x1] = slice_in[pos_in + 0x1]

        # Right half
        pos_in = ipair * pair_len + run_len + idx % run_len

        if pos_in < len
            lo = ipair * pair_len
            hi = lo + run_len
            pos_out = pos_in - run_len + upper_bound0(slice_in, slice_in[pos_in + 0x1], lo, hi, comp) - lo
            slice_out[pos_out + 0x1] = slice_in[pos_in + 0x1]
        end
    end
end


# GPU merge sort of `v` (or of each slice along `dims`), in place, with the scratch buffers of
# `sort_sizes(::MergeSort, ...)`.
function merge_sort!(
    v::AbstractArray, backend::Backend, bufs::NamedTuple, ord::Base.Order.Ordering, by;
    block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return v

    # Compute keys once instead of evaluating `by` in every comparison, into an array of the
    # keys' type (broadcasting would make a `BitArray` of `Bool` keys on the host)
    if by !== identity
        keys = bufs.keys
        map!(by, keys, v; backend, block_size)
        merge_sort_by_key!(keys, v, backend, bufs, ord.order; block_size, dims)
        return v
    end

    # Construct comparator
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level: each block sorts a tile of one slice in local memory
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    merge_tile_kernel!(backend, block_size)(
        v, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level: merge the sorted tiles of each slice, doubling the run length every pass
    # (an `Int`: in slices of more than 2^30 elements, the run length doubles past `typemax(Int32)`)
    run_len = block_size * 2
    if len > run_len
        p1 = v
        p2 = bufs.temp

        kernel! = merge_pass_kernel!(backend, block_size)

        niter = 0
        while len > run_len
            blocks = ((len + run_len - 1) ÷ run_len + 1) ÷ 2 * (run_len ÷ block_size)
            kernel!(
                p1, p2, comp, run_len, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            run_len = run_len << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto_on!(v, p1, backend)
        end
    end

    v
end
