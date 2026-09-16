@kernel inbounds=true cpu=false unsafe_indices=true function _merge_sort_block!(
    vec, comp, layout, blocks_per_slice,
)

    @uniform N = @groupsize()[1]
    s_buf = @localmem eltype(vec) (N * 0x2,)

    T = eltype(vec)
    I = typeof(N)

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block sorts a tile of one slice
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    elems = slice(vec, layout, islice)
    len = layout.len

    i = ithread + iblock * N * 0x2
    i < len && (s_buf[ithread + 0x1] = elems[i + 0x1])

    i = ithread + N + iblock * N * 0x2
    i < len && (s_buf[ithread + N + 0x1] = elems[i + 0x1])

    @synchronize()

    half_size_group = typeof(ithread)(1)
    size_group = typeof(ithread)(2)

    while half_size_group <= N
        gid = ithread ÷ half_size_group

        local v1::T
        local v2::T
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = gid * size_group + half_size_group + iblock * N * 0x2
        if i < len
            tid = gid * size_group + ithread % half_size_group
            v1 = s_buf[tid + 0x1]

            i = (gid + 0x1) * size_group + iblock * N * 0x2
            n = i < len ? half_size_group : len - iblock * N * 0x2 - gid * size_group - half_size_group
            lo = gid * size_group + half_size_group
            hi = lo + n
            pos1 = ithread % half_size_group + _lower_bound_s0(s_buf, v1, lo, hi, comp) - lo
        end

        tid = gid * size_group + half_size_group + ithread % half_size_group
        i = tid + iblock * N * 0x2
        if i < len
            v2 = s_buf[tid + 0x1]
            lo = gid * size_group
            hi = lo + half_size_group
            pos2 = ithread % half_size_group + _upper_bound_s0(s_buf, v2, lo, hi, comp) - lo
        end

        @synchronize()

        pos1 != typemax(I) && (s_buf[gid * size_group + pos1 + 0x1] = v1)
        pos2 != typemax(I) && (s_buf[gid * size_group + pos2 + 0x1] = v2)

        @synchronize()

        half_size_group = half_size_group << 0x1
        size_group = size_group << 0x1
    end

    i = ithread + iblock * N * 0x2
    i < len && (elems[i + 0x1] = s_buf[ithread + 0x1])

    i = ithread + N + iblock * N * 0x2
    i < len && (elems[i + 0x1] = s_buf[ithread + N + 0x1])
end


@kernel inbounds=true cpu=false unsafe_indices=true function _merge_sort_global!(
    @Const(vec_in), vec_out, comp, half_size_group, layout, blocks_per_slice,
)
    N = @groupsize()[1]

    # Use zero-based indices internally and half-open search bounds.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Merges never cross slice boundaries
    islice, iblock = slice_block(layout, iblock, blocks_per_slice)
    slice_in = slice(vec_in, layout, islice)
    slice_out = slice(vec_out, layout, islice)
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
        pos_in < len && (slice_out[pos_in + 0x1] = slice_in[pos_in + 0x1])
    else

        hi = (gid + 0x1) * size_group
        hi > len && (hi = len)

        pos_out = pos_in + _lower_bound_s0(slice_in, slice_in[pos_in + 0x1], lo, hi, comp) - lo
        slice_out[pos_out + 0x1] = slice_in[pos_in + 0x1]

        # Right half
        pos_in = gid * size_group + half_size_group + idx % half_size_group

        if pos_in < len
            lo = gid * size_group
            hi = lo + half_size_group
            pos_out = pos_in - half_size_group + _upper_bound_s0(slice_in, slice_in[pos_in + 0x1], lo, hi, comp) - lo
            slice_out[pos_out + 0x1] = slice_in[pos_in + 0x1]
        end
    end
end


"""
    merge_sort!(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,

        # Sort each 1D slice along this dimension; `:` sorts the whole array as one vector
        dims::Union{Colon, Integer}=Colon(),
    )
"""
function merge_sort!(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    dims::Union{Colon, Integer}=Colon(),
)
    # Simple sanity checks
    @argcheck block_size > 0
    layout = slice_layout(v, dims)
    ord = Base.Order.ord(lt, by, rev, order)
    if !isnothing(temp)
        @argcheck length(temp) == length(v)
        @argcheck eltype(temp) === eltype(v)
    end
    (isempty(v) || layout.len <= 1) && return v

    # Compute keys once instead of evaluating `by` in every comparison.
    if by !== identity
        keys = by.(v)
        merge_sort_by_key!(
            keys, v, backend;
            lt, rev, order, block_size, dims,
            temp_values=temp,   # temp was for v swap buffer; maps to temp_values here
        )
        return v
    end

    # Construct comparator
    comp = (x, y) -> Base.Order.lt(ord, x, y)

    # Block level: each block sorts a tile of one slice in local memory
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_block!(backend, block_size)(
        v, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level: merge the sorted tiles of each slice, doubling the run length every pass
    half_size_group = Int32(block_size * 2)
    size_group = half_size_group * 2
    if len > half_size_group
        p1 = v
        p2 = isnothing(temp) ? similar(v) : temp

        kernel! = _merge_sort_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(
                p1, p2, comp, half_size_group, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto!(v, p1)
        end
    end

    v
end


"""
    merge_sort(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        dims::Union{Colon, Integer}=Colon(),
    )
"""
function merge_sort(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    v_copy = copy(v)
    merge_sort!(
        v_copy, backend;
        kwargs...
    )
end
