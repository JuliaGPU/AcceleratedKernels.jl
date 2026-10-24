@kernel inbounds=true cpu=false unsafe_indices=true function merge_tile_by_key_kernel!(
    keys, values, comp, layout, blocks_per_slice,
)

    @uniform N = @groupsize()[1]
    s_keys = @localmem eltype(keys) (N * 0x2,)
    s_values = @localmem eltype(values) (N * 0x2,)

    I = typeof(N)

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block sorts a tile of one slice
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    slice_keys = slice(keys, layout, islice)
    slice_values = slice(values, layout, islice)
    len = layout.len

    i = ithread + iblock * N * 0x2
    if i < len
        s_keys[ithread + 0x1] = slice_keys[i + 0x1]
        s_values[ithread + 0x1] = slice_values[i + 0x1]
    end

    i = ithread + N + iblock * N * 0x2
    if i < len
        s_keys[ithread + N + 0x1] = slice_keys[i + 0x1]
        s_values[ithread + N + 0x1] = slice_values[i + 0x1]
    end

    @synchronize()

    run_len = typeof(ithread)(1)
    pair_len = typeof(ithread)(2)

    while run_len <= N
        ipair = ithread ÷ run_len

        local k1::eltype(keys)
        local k2::eltype(keys)
        local v1::eltype(values)
        local v2::eltype(values)
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = ipair * pair_len + run_len + iblock * N * 0x2
        if i < len
            tid = ipair * pair_len + ithread % run_len
            k1 = s_keys[tid + 0x1]
            v1 = s_values[tid + 0x1]

            i = (ipair + 0x1) * pair_len + iblock * N * 0x2
            n = i < len ? run_len : len - iblock * N * 0x2 - ipair * pair_len - run_len
            lo = ipair * pair_len + run_len
            hi = lo + n
            pos1 = ithread % run_len + lower_bound0(s_keys, k1, lo, hi, comp) - lo
        end

        tid = ipair * pair_len + run_len + ithread % run_len
        i = tid + iblock * N * 0x2
        if i < len
            k2 = s_keys[tid + 0x1]
            v2 = s_values[tid + 0x1]
            lo = ipair * pair_len
            hi = lo + run_len
            pos2 = ithread % run_len + upper_bound0(s_keys, k2, lo, hi, comp) - lo
        end

        @synchronize()

        if pos1 != typemax(I)
            s_keys[ipair * pair_len + pos1 + 0x1] = k1
            s_values[ipair * pair_len + pos1 + 0x1] = v1
        end
        if pos2 != typemax(I)
            s_keys[ipair * pair_len + pos2 + 0x1] = k2
            s_values[ipair * pair_len + pos2 + 0x1] = v2
        end

        @synchronize()

        run_len = run_len << 0x1
        pair_len = pair_len << 0x1
    end

    i = ithread + iblock * N * 0x2
    if i < len
        slice_keys[i + 0x1] = s_keys[ithread + 0x1]
        slice_values[i + 0x1] = s_values[ithread + 0x1]
    end

    i = ithread + N + iblock * N * 0x2
    if i < len
        slice_keys[i + 0x1] = s_keys[ithread + N + 0x1]
        slice_values[i + 0x1] = s_values[ithread + N + 0x1]
    end
end


@kernel inbounds=true cpu=false unsafe_indices=true function merge_pass_by_key_kernel!(
    @Const(keys_in), keys_out,
    @Const(values_in), values_out,
    comp, run_len, layout, blocks_per_slice,
)

    N = @groupsize()[1]

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Merges never cross slice boundaries
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    keys_in = slice(keys_in, layout, islice)
    keys_out = slice(keys_out, layout, islice)
    values_in = slice(values_in, layout, islice)
    values_out = slice(values_out, layout, islice)
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
        if pos_in < len
            keys_out[pos_in + 0x1] = keys_in[pos_in + 0x1]
            values_out[pos_in + 0x1] = values_in[pos_in + 0x1]
        end
    else

        hi = (ipair + 0x1) * pair_len
        hi > len && (hi = len)

        pos_out = pos_in + lower_bound0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
        keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
        values_out[pos_out + 0x1] = values_in[pos_in + 0x1]

        # Right half
        pos_in = ipair * pair_len + run_len + idx % run_len

        if pos_in < len
            lo = ipair * pair_len
            hi = lo + run_len
            pos_out = pos_in - run_len + upper_bound0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
            keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
            values_out[pos_out + 0x1] = values_in[pos_in + 0x1]
        end
    end
end


# GPU merge sort of `keys` (or of each slice along `dims`), permuting `values` alike.
# GPU merge sort of `keys` (or of each slice along `dims`) carrying `values` along, in place, with
# the scratch buffers of `merge_by_key_scratch`.
function merge_sort_by_key!(
    keys::AbstractArray,
    values::AbstractArray,
    backend::Backend,
    bufs::NamedTuple;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    dims::Union{Colon, Integer}=Colon(),
)
    # Simple sanity checks
    @argcheck length(keys) == length(values)
    layout = slice_layout(keys, dims)
    if !(dims isa Colon)
        @argcheck axes(keys) == axes(values)
    end

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev, order)
    (isempty(keys) || layout.len <= 1) && return keys, values
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level: each block sorts a tile of one slice in local memory
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    merge_tile_by_key_kernel!(backend, block_size)(
        keys, values, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level: merge the sorted tiles of each slice, doubling the run length every pass
    # (an `Int`: in slices of more than 2^30 elements, the run length doubles past `typemax(Int32)`)
    run_len = block_size * 2
    if len > run_len
        pk1 = keys
        pk2 = bufs.temp_keys

        pv1 = values
        pv2 = bufs.temp_values

        kernel! = merge_pass_by_key_kernel!(backend, block_size)

        niter = 0
        while len > run_len
            blocks = ((len + run_len - 1) ÷ run_len + 1) ÷ 2 * (run_len ÷ block_size)
            kernel!(
                pk1, pk2, pv1, pv2, comp, run_len, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            run_len = run_len << 1;
            pk1, pk2 = pk2, pk1
            pv1, pv2 = pv2, pv1

            niter += 1
        end

        if isodd(niter)
            copyto_on!(keys, pk1, backend)
            copyto_on!(values, pv1, backend)
        end
    end

    keys, values
end


# The scratch of `merge_sort_by_key!` over slices `layout` of keys of size `ksize` and type `K`,
# with values of size `vsize` and type `V`: swap buffers, if the tiles need merging
merge_by_key_scratch(a, layout, ksize, ::Type{K}, vsize, ::Type{V}) where {K, V} =
    layout.len > 2 * a.block_size ?
        (; temp_keys=scratch(K, ksize), temp_values=scratch(V, vsize)) : (;)
