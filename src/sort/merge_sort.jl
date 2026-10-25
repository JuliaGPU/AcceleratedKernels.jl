# Stable GPU merge sort of `keys` under a comparator `comp`, moving `values` along unless they are
# `nothing`. Each block sorts a tile of `2 * block_size` elements of one slice in local memory;
# global passes then merge pairs of sorted runs, doubling the run length. In both, every element
# finds its destination by binary search in the run it merges with.

Base.@propagate_inbounds getvalue(values, i) = values[i]
getvalue(::Nothing, i) = nothing
Base.@propagate_inbounds setvalue!(values, i, x) = (values[i] = x; nothing)
setvalue!(::Nothing, i, x) = nothing


@kernel inbounds=true cpu=false unsafe_indices=true function merge_tile_kernel!(
    keys, values, comp, layout, blocks_per_slice,
)

    @uniform N = @groupsize()[1]
    s_keys = @localmem eltype(keys) (N * 0x2,)
    s_values = values === nothing ? nothing : @localmem eltype(values) (N * 0x2,)

    K = eltype(keys)
    V = values === nothing ? Nothing : eltype(values)
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
        setvalue!(s_values, ithread + 0x1, getvalue(slice_values, i + 0x1))
    end

    i = ithread + N + iblock * N * 0x2
    if i < len
        s_keys[ithread + N + 0x1] = slice_keys[i + 0x1]
        setvalue!(s_values, ithread + N + 0x1, getvalue(slice_values, i + 0x1))
    end

    @synchronize()

    run_len = typeof(ithread)(1)
    pair_len = typeof(ithread)(2)

    while run_len <= N
        ipair = ithread ÷ run_len

        local k1::K
        local k2::K
        local v1::V
        local v2::V
        pos1 = typemax(I)
        pos2 = typemax(I)

        i = ipair * pair_len + run_len + iblock * N * 0x2
        if i < len
            tid = ipair * pair_len + ithread % run_len
            k1 = s_keys[tid + 0x1]
            v1 = getvalue(s_values, tid + 0x1)

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
            v2 = getvalue(s_values, tid + 0x1)
            lo = ipair * pair_len
            hi = lo + run_len
            pos2 = ithread % run_len + upper_bound0(s_keys, k2, lo, hi, comp) - lo
        end

        @synchronize()

        if pos1 != typemax(I)
            s_keys[ipair * pair_len + pos1 + 0x1] = k1
            setvalue!(s_values, ipair * pair_len + pos1 + 0x1, v1)
        end
        if pos2 != typemax(I)
            s_keys[ipair * pair_len + pos2 + 0x1] = k2
            setvalue!(s_values, ipair * pair_len + pos2 + 0x1, v2)
        end

        @synchronize()

        run_len = run_len << 0x1
        pair_len = pair_len << 0x1
    end

    i = ithread + iblock * N * 0x2
    if i < len
        slice_keys[i + 0x1] = s_keys[ithread + 0x1]
        setvalue!(slice_values, i + 0x1, getvalue(s_values, ithread + 0x1))
    end

    i = ithread + N + iblock * N * 0x2
    if i < len
        slice_keys[i + 0x1] = s_keys[ithread + N + 0x1]
        setvalue!(slice_values, i + 0x1, getvalue(s_values, ithread + N + 0x1))
    end
end


@kernel inbounds=true cpu=false unsafe_indices=true function merge_pass_kernel!(
    @Const(keys_in), keys_out, @Const(values_in), values_out,
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
            setvalue!(values_out, pos_in + 0x1, getvalue(values_in, pos_in + 0x1))
        end
    else

        hi = (ipair + 0x1) * pair_len
        hi > len && (hi = len)

        pos_out = pos_in + lower_bound0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
        keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
        setvalue!(values_out, pos_out + 0x1, getvalue(values_in, pos_in + 0x1))

        # Right half
        pos_in = ipair * pair_len + run_len + idx % run_len

        if pos_in < len
            lo = ipair * pair_len
            hi = lo + run_len
            pos_out = pos_in - run_len + upper_bound0(keys_in, keys_in[pos_in + 0x1], lo, hi, comp) - lo
            keys_out[pos_out + 0x1] = keys_in[pos_in + 0x1]
            setvalue!(values_out, pos_out + 0x1, getvalue(values_in, pos_in + 0x1))
        end
    end
end


# Sort the slices `layout` of `keys` in place under `comp`, moving `values` (or `nothing`) along.
# `temp_keys` and `temp_values` are swap buffers like `keys` and `values`, used only when the slices
# are longer than a tile (`nothing` otherwise).
function merge_sort_slices!(keys, values, temp_keys, temp_values, comp, layout, backend, block_size)
    # Block level: each block sorts a tile of one slice in local memory
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    merge_tile_kernel!(backend, block_size)(
        keys, values, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level: merge the sorted tiles of each slice, doubling the run length every pass
    # (an `Int`: in slices of more than 2^30 elements, the run length doubles past `typemax(Int32)`)
    run_len = block_size * 2
    len > run_len || return
    k1, k2 = keys, temp_keys
    v1, v2 = values, temp_values
    kernel! = merge_pass_kernel!(backend, block_size)
    niter = 0
    while len > run_len
        blocks = ((len + run_len - 1) ÷ run_len + 1) ÷ 2 * (run_len ÷ block_size)
        kernel!(
            k1, k2, v1, v2, comp, run_len, layout, blocks,
            ndrange=(block_size * blocks * slice_count(layout),),
        )
        run_len = run_len << 1
        k1, k2 = k2, k1
        v1, v2 = v2, v1
        niter += 1
    end
    if isodd(niter)
        copyto_on!(keys, k1, backend)
        values === nothing || copyto_on!(values, v1, backend)
    end
    return
end


# GPU merge sort of `v` (or of each slice along `dims`), in place, with the scratch buffers of
# `sort_plan(::MergeSort, ...)`.
function merge_sort!(
    v::AbstractArray, backend::Backend, bufs::NamedTuple, ord::Base.Order.Ordering, by;
    block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return v

    # Compute keys once instead of evaluating `by` in every comparison, into an array of the
    # keys' type (broadcasting would make a `BitArray` of `Bool` keys on the host)
    if by !== identity
        keys = map!(by, bufs.keys, v; backend, block_size)
        merge_sort_by_key!(keys, v, backend, bufs, ord.order; block_size, dims)
        return v
    end

    comp = (x, y) -> Base.Order.lt(ord, x, y)
    merge_sort_slices!(v, nothing, get(bufs, :temp, nothing), nothing, comp, layout, backend,
                       block_size)
    v
end


# GPU merge sort of `keys` (or of each slice along `dims`) carrying `values` along, in place, with
# the scratch buffers of `merge_by_key_scratch`.
function merge_sort_by_key!(
    keys::AbstractArray, values::AbstractArray, backend::Backend, bufs::NamedTuple,
    ord::Base.Order.Ordering; block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(keys, dims)
    (isempty(keys) || layout.len <= 1) && return keys, values
    comp = (x, y) -> Base.Order.lt(ord, x, y)
    merge_sort_slices!(keys, values, get(bufs, :temp_keys, nothing),
                       get(bufs, :temp_values, nothing), comp, layout, backend, block_size)
    keys, values
end


# The scratch of `merge_sort_by_key!` over slices `layout` of keys of size `ksize` and type `K`,
# with values of size `vsize` and type `V`: swap buffers, if the tiles need merging
merge_by_key_scratch(a, layout, ksize, ::Type{K}, vsize, ::Type{V}) where {K, V} =
    layout.len > 2 * a.block_size ?
        (; temp_keys=scratch(K, ksize), temp_values=scratch(V, vsize)) : (;)


# GPU merge sort permutation: sorts a copy of the keys, `by.(v)`, carrying the indices along.
function merge_sortperm!(
    ix::AbstractArray, v::AbstractArray, backend::Backend, bufs::NamedTuple,
    ord::Base.Order.Ordering, by; block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex_on(Base.OneTo(length(ix)), backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(v) || layout.len <= 1) && return ix
    if by === identity
        # (a range, for instance, has no backend: collected on the host first)
        keys = copyto_on!(bufs.keys, backendless(v) ? collect(v) : v, backend)
        merge_sort_by_key!(keys, ix, backend, bufs, ord; block_size, dims)
    else
        # Evaluated once instead of in every comparison; `ord` is `By(by, ord.order)`
        keys = map!(by, bufs.keys, v; backend, block_size)
        merge_sort_by_key!(keys, ix, backend, bufs, ord.order; block_size, dims)
    end

    ix
end


# GPU merge sort permutation comparing the keys in global memory, without copying them: the
# indices are the keys, compared through `v`.
function merge_sortperm_lowmem!(
    ix::AbstractArray, v::AbstractArray, backend::Backend, bufs::NamedTuple,
    ord::Base.Order.Ordering; block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex_on(Base.OneTo(length(ix)), backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(ix) || layout.len <= 1) && return ix

    comp = (ix, iy) -> Base.Order.lt(ord, v[ix], v[iy])
    merge_sort_slices!(ix, nothing, get(bufs, :temp, nothing), nothing, comp, layout, backend,
                       block_size)
    ix
end
