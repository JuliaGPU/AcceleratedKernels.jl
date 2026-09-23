# Segmented radix sort: the `sort(A; dims=1)` path for `RadixSort`. Each column is one segment
# (contiguous in column-major order); all segments are sorted together with an LSD radix whose
# histogram is laid out segment-major, so a plain exclusive scan gives `seg*L + within-segment base`
# (each earlier segment contributes exactly `L` elements). Reuses the flat radix key transform.

# Tuned defaults: ~13 KB of shared memory, sized to fit every backend's local-memory budget while
# staying near the fastest config across a block-size / items / chunk-width sweep.
const _SEG_BLOCK = 256   # threads per block
const _SEG_ITEMS = 2     # elements per thread
const _SEG_CHUNK = 64    # rank-chunk width

# Per-block digit histogram, written segment-major: hist[seg*256*bps + digit*bps + blk].
@kernel inbounds=true cpu=false unsafe_indices=true function _seg_radix_hist!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val{ITEMS}, L::Int, bps::Int,
) where ITEMS
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (256,)
    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    seg = iblock ÷ bps
    blk = iblock % bps

    j = ithread
    while j < 256; s_hist[j + 1] = UInt32(0); j += NI; end
    @synchronize()

    m = 0
    while m < ITEMS
        lp = blk * NI * ITEMS + ithread + m * NI
        if lp < L
            d = Int(_rs_digit(v[seg * L + lp + 1], shift, rev))
            Atomix.@atomic s_hist[d + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    base = seg * 256 * bps + blk
    b = ithread
    while b < 256; hist[base + b * bps + 1] = s_hist[b + 1]; b += NI; end
end

# Stable scatter: the scanned histogram gives each (block, digit) its global base; a chunked rank
# (same-digit predecessors within the chunk, plus earlier chunks) places each element within it.
@kernel inbounds=true cpu=false unsafe_indices=true function _seg_radix_scatter!(
    vout, @Const(vin), @Const(hist), shift::UInt32, rev::Bool, ::Val{ITEMS}, ::Val{CH}, L::Int, bps::Int,
) where {ITEMS, CH}
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * ITEMS
    @uniform NCH  = (Int(@groupsize()[1]) * ITEMS) ÷ CH
    s_elem  = @localmem eltype(vin) (TILE,)
    s_digit = @localmem UInt32       (TILE,)
    s_gbase = @localmem UInt32       (256,)
    s_chist = @localmem UInt32       (256 * NCH,)
    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    seg = iblock ÷ bps
    blk = iblock % bps

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        lp = blk * TILE + p
        if lp < L
            k = vin[seg * L + lp + 1]
            s_elem[p + 1]  = k
            s_digit[p + 1] = UInt32(_rs_digit(k, shift, rev))
        else
            s_digit[p + 1] = 0xffffffff
        end
        m += 1
    end
    base = seg * 256 * bps + blk
    j = ithread
    while j < 256; s_gbase[j + 1] = hist[base + j * bps + 1]; j += NI; end
    j = ithread
    while j < 256 * NCH; s_chist[j + 1] = UInt32(0); j += NI; end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            Atomix.@atomic s_chist[(p ÷ CH) * 256 + Int(d) + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    d = ithread
    while d < 256
        acc = UInt32(0)
        c = 0
        while c < NCH
            cnt = s_chist[c * 256 + d + 1]
            s_chist[c * 256 + d + 1] = acc
            acc += cnt
            c += 1
        end
        d += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            cs = (p ÷ CH) * CH
            cnt = UInt32(0)
            for r in 0:CH - 1
                q = cs + r
                cnt += UInt32((q < p) & (s_digit[q + 1] == d))
            end
            gpos = Int(s_gbase[Int(d) + 1]) + Int(s_chist[(p ÷ CH) * 256 + Int(d) + 1] + cnt)
            vout[gpos + 1] = s_elem[p + 1]
        end
        m += 1
    end
end

"""
    _segmented_radix_sort_dims!(v, backend; descending)

Sort each 1-D slice of `v` along dimension 1 with a segment-major LSD radix. In-place; `v` may be any
`N`-dimensional array (its first dimension is the slice). Supports the same bits eltypes as the flat
radix, forward or reverse ordering.
"""
function _segmented_radix_sort_dims!(
    v::AbstractArray{T}, backend::Backend=get_backend(v);
    descending::Bool=false,
) where T
    L = size(v, 1)
    n = length(v)
    S = n ÷ L
    TILE = _SEG_BLOCK * _SEG_ITEMS
    bps = cld(L, TILE)
    nblocks = S * bps

    vflat = reshape(v, n)
    hist = similar(vflat, UInt32, 256 * nblocks)
    p1 = vflat
    p2 = similar(vflat)
    npass = sizeof(T) * 8 ÷ 8
    ndrange = _SEG_BLOCK * nblocks

    for pass in 0:npass - 1
        shift = UInt32(pass * 8)
        _seg_radix_hist!(backend, _SEG_BLOCK)(
            hist, p1, shift, descending, Val(_SEG_ITEMS), L, bps; ndrange)
        accumulate!(+, hist, backend; init=UInt32(0), inclusive=false)
        _seg_radix_scatter!(backend, _SEG_BLOCK)(
            p2, p1, hist, shift, descending, Val(_SEG_ITEMS), Val(_SEG_CHUNK), L, bps; ndrange)
        p1, p2 = p2, p1
    end

    KernelAbstractions.synchronize(backend)
    p1 !== vflat && copyto!(vflat, p1)
    v
end
