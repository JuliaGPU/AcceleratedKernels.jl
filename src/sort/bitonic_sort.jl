# Bitonic sort with every comparator ascending in `ord`. At each level `kk = 2, 4, ..., N`,
# the first step pairs zero-based `i` with `i ⊻ (kk - 1)`, reflecting the second ascending run.
# The remaining strides `kk/4, ..., 1` merge the two runs without sorting either descending.
#
# For other lengths, imagine padding with values ordered after all real elements. Ascending
# comparators leave that padding at the end, so we skip pairs whose partner is past the length.
# No padding storage or sentinel value is needed.
#
# Strides below the tile span run in local memory; larger strides each need a global pass.
# Both kernels use slices.jl layouts, with one flat slice for whole-array sorts.

# Elements `i` and `partner` (zero-based) of comparator `p` at level `kklog`, stride `jlog`
@inline function bitonic_pair(p, kklog, jlog)
    j = 1 << jlog
    i = ((p >> jlog) << (jlog + 1)) | (p & (j - 1))
    partner = jlog == kklog - 1 ? i ⊻ (2j - 1) : i + j
    i, partner
end

Base.@propagate_inbounds function compare_exchange!(elems, ord, i, partner)
    a = elems[i + 1]
    b = elems[partner + 1]
    if Base.Order.lt(ord, b, a)
        elems[i + 1] = b
        elems[partner + 1] = a
    end
end


# Run the local strides of levels `first_level:last_level`. A tile holds part of one long slice
# (`SPAN == CAP`) or `CAP ÷ SPAN` short slices. Unused entries are never read.
@kernel cpu=false inbounds=true unsafe_indices=true function bitonic_tile_kernel!(
    v, ord, layout, tiles_per_slice::Int, first_level::Int, last_level::Int, ::Val{CAP}, ::Val{SPAN},
) where {CAP, SPAN}
    @uniform BS = Int(@groupsize()[1])
    tile = @localmem eltype(v) (CAP,)

    iblock = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, itile = slice_block(layout, iblock, tiles_per_slice)
    islice *= CAP ÷ SPAN
    nslices = slice_count(layout)
    spanlog = trailing_zeros(SPAN)
    base = itile * SPAN
    len = layout.len - base

    # Packing arithmetic folds away when SPAN == CAP.
    pos = ithread
    while pos < CAP
        s = SPAN == CAP ? islice : islice + (pos >> spanlog)
        off = SPAN == CAP ? pos : pos & (SPAN - 1)
        if off < len && s < nslices
            tile[pos + 1] = slice(v, layout, s)[base + off + 1]
        end
        pos += BS
    end
    @synchronize()

    kklog = first_level
    while kklog <= last_level
        jlog = min(kklog - 1, spanlog - 1)
        while jlog >= 0
            p = ithread
            while p < CAP ÷ 2
                offset = SPAN == CAP ? 0 : (p >> (spanlog - 1)) * SPAN
                i, partner = bitonic_pair(SPAN == CAP ? p : p & (SPAN ÷ 2 - 1), kklog, jlog)
                if partner < len && (SPAN == CAP || islice + offset ÷ SPAN < nslices)
                    compare_exchange!(tile, ord, offset + i, offset + partner)
                end
                p += BS
            end
            @synchronize()
            jlog -= 1
        end
        kklog += 1
    end

    pos = ithread
    while pos < CAP
        s = SPAN == CAP ? islice : islice + (pos >> spanlog)
        off = SPAN == CAP ? pos : pos & (SPAN - 1)
        if off < len && s < nslices
            slice(v, layout, s)[base + off + 1] = tile[pos + 1]
        end
        pos += BS
    end
end


# One global step at stride `2^jlog >= CAP`, one thread per comparator
@kernel cpu=false inbounds=true function bitonic_pass_kernel!(
    v, ord, layout, pairs_per_slice, kklog::Int, jlog::Int,
)
    islice, p = slice_block(layout, Int(@index(Global, Linear)) - 1, pairs_per_slice)
    elems = slice(v, layout, islice)
    i, partner = bitonic_pair(p, kklog, jlog)
    partner < layout.len && compare_exchange!(elems, ord, i, partner)
end


# GPU bitonic sort of `v` (or of each slice along `dims`), in place.
function bitonic_sort!(
    v::AbstractArray, backend::Backend, ord::Base.Order.Ordering;
    block_size::Int, items_per_thread::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)
    len = layout.len
    (isempty(v) || len <= 1) && return v

    # Pack short slices into one tile; split long slices across tiles.
    tile_size = block_size * items_per_thread
    npow = nextpow(2, len)
    nslices = slice_count(layout)
    span = min(npow, tile_size)
    cap = span < block_size ? span * nextpow(2, min(nslices, tile_size ÷ span)) : span
    tiles_per_slice = cld(len, span)
    nblocks = cld(nslices, cap ÷ span) * tiles_per_slice
    tile_threads = min(block_size, cap)
    tile! = bitonic_tile_kernel!(backend, tile_threads)
    tile!(v, ord, layout, tiles_per_slice, 1, trailing_zeros(span), Val(cap), Val(span);
          ndrange=nblocks * tile_threads)

    if npow > span
        global! = bitonic_pass_kernel!(backend, block_size)
        pairs = npow ÷ 2
        for kklog in (trailing_zeros(span) + 1):trailing_zeros(npow)
            for jlog in (kklog - 1):-1:trailing_zeros(span)
                global!(v, ord, layout, pairs, kklog, jlog; ndrange=nslices * pairs)
            end
            tile!(v, ord, layout, tiles_per_slice, kklog, kklog, Val(cap), Val(span);
                  ndrange=nblocks * tile_threads)
        end
    end

    v
end
