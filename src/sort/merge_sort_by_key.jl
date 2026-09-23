@kernel inbounds=true cpu=false unsafe_indices=true function _merge_sort_by_key_block!(
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

    half_size_group = typeof(ithread)(1)
    size_group = typeof(ithread)(2)

    while half_size_group <= N
        gid = ithread ÷ half_size_group

        local k1::eltype(keys)
        local k2::eltype(keys)
        local v1::eltype(values)
        local v2::eltype(values)
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = gid * size_group + half_size_group + iblock * N * 0x2
        if i < len
            tid = gid * size_group + ithread % half_size_group
            k1 = s_keys[tid + 0x1]
            v1 = s_values[tid + 0x1]

            i = (gid + 0x1) * size_group + iblock * N * 0x2
            n = i < len ? half_size_group : len - iblock * N * 0x2 - gid * size_group - half_size_group
            lo = gid * size_group + half_size_group
            hi = lo + n
            pos1 = ithread % half_size_group + _lower_bound_s0(s_keys, k1, lo, hi, comp) - lo
        end

        tid = gid * size_group + half_size_group + ithread % half_size_group
        i = tid + iblock * N * 0x2
        if i < len
            k2 = s_keys[tid + 0x1]
            v2 = s_values[tid + 0x1]
            lo = gid * size_group
            hi = lo + half_size_group
            pos2 = ithread % half_size_group + _upper_bound_s0(s_keys, k2, lo, hi, comp) - lo
        end

        @synchronize()

        if pos1 != typemax(I)
            s_keys[gid * size_group + pos1 + 0x1] = k1
            s_values[gid * size_group + pos1 + 0x1] = v1
        end
        if pos2 != typemax(I)
            s_keys[gid * size_group + pos2 + 0x1] = k2
            s_values[gid * size_group + pos2 + 0x1] = v2
        end

        @synchronize()

        half_size_group = half_size_group << 0x1
        size_group = size_group << 0x1
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


@kernel inbounds=true cpu=false unsafe_indices=true function _merge_sort_by_key_global!(
    keys_in_arg, keys_out,
    values_in_arg, values_out,
    comp, half_size_group, layout, blocks_per_slice,
)
    keys_in = _const_source(keys_in_arg)
    values_in = _const_source(values_in_arg)

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
    size_group = half_size_group * 0x2
    gid = idx ÷ half_size_group

    # Left half
    pos_in = gid * size_group + idx % half_size_group
    lo = gid * size_group + half_size_group

    if lo >= len
        # Incomplete left half, nothing to swap on the right, simply copy elements to be sorted
        # in next iteration
        if pos_in < len
            keys_out[pos_in + 0x1] = keys_in[pos_in + 0x1]
            values_out[pos_in + 0x1] = values_in[pos_in + 0x1]
        end
    else

        hi = (gid + 0x1) * size_group
        hi > len && (hi = len)

        pos_out = pos_in + _lower_bound_s0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
        keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
        values_out[pos_out + 0x1] = values_in[pos_in + 0x1]

        # Right half
        pos_in = gid * size_group + half_size_group + idx % half_size_group

        if pos_in < len
            lo = gid * size_group
            hi = lo + half_size_group
            pos_out = pos_in - half_size_group + _upper_bound_s0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
            keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
            values_out[pos_out + 0x1] = values_in[pos_in + 0x1]
        end
    end
end


# GPU merge sort of `keys` (or of each slice along `dims`), permuting `values` alike.
function _merge_sort_by_key!(
    keys::AbstractArray,
    values::AbstractArray,
    backend::Backend=get_backend(keys);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp_keys::Union{Nothing, AbstractArray}=nothing,
    temp_values::Union{Nothing, AbstractArray}=nothing,
    dims::Union{Colon, Integer}=Colon(),
)
    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(keys) == length(values)
    layout = slice_layout(keys, dims)
    if !(dims isa Colon)
        @argcheck axes(keys) == axes(values)
    end
    if !isnothing(temp_keys)
        @argcheck length(temp_keys) == length(keys)
        @argcheck eltype(temp_keys) === eltype(keys)
    end
    if !isnothing(temp_values)
        @argcheck length(temp_values) == length(values)
        @argcheck eltype(temp_values) === eltype(values)
    end

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev, order)
    (isempty(keys) || layout.len <= 1) && return keys, values
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level: each block sorts a tile of one slice in local memory
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_by_key_block!(backend, block_size)(
        keys, values, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level: merge the sorted tiles of each slice, doubling the run length every pass
    half_size_group = Int32(block_size * 2)
    size_group = half_size_group * 2
    if len > half_size_group
        pk1 = keys
        pk2 = isnothing(temp_keys) ? similar(keys) : temp_keys

        pv1 = values
        pv2 = isnothing(temp_values) ? similar(values) : temp_values

        kernel! = _merge_sort_by_key_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(
                pk1, pk2, pv1, pv2, comp, half_size_group, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            pk1, pk2 = pk2, pk1
            pv1, pv2 = pv2, pv1

            niter += 1
        end

        if isodd(niter)
            copyto!(keys, pk1)
            copyto!(values, pv1)
        end
    end

    keys, values
end
