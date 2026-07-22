# LSD radix sort (8-bit, 256 buckets).  Stable, GPU-only.
#
# Backend-portable: no dependency on sub-group / warp intrinsics.  Where a
# backend reports shared-memory atomics support, faster atomic-based histogram
# and scatter kernels are used; otherwise a scan-based path runs anywhere.
#
# Algorithm (per non-trivial pass over byte k):
#   1. histogram — count, per block, how many elements have each byte-digit.
#                  Layout: hist[k * B + b] = count of digit k in block b; B = num_blocks.
#                  Shared-atomic where supported (_radix_hist_atomic!), portable
#                  per-bucket scan otherwise (_radix_hist!).
#   2. accumulate! — exclusive prefix sum over hist → global per-(digit, block) offsets.
#   3. scatter — stable scatter to the offsets.  Chunked O(32)-rank where shared-memory
#                atomics are available (_radix_scatter_chunked!), O(block_size)-rank
#                broadcast scan otherwise (_radix_scatter!).
#
# Key optimizations:
#   • Fused min/max range (_rs_key_range): one reduction over the sort keys instead of
#     separate minimum + maximum.
#   • Items-per-thread tiling: the fast kernels process items_per_thread elements
#     per thread, shrinking the per-(digit, block) histogram — and the exclusive
#     scan over it — by the same factor.
#   • Skip-pass via min/max keys: if min and max share the whole byte-suffix from byte k
#     up, every element does too, so the whole pass is skipped (e.g. UInt32 in [0, 255]
#     sorts in a single pass).
#
# Supported element types: UInt32/64, Int32/64, Float32/64.
# Custom lt/by → falls back to merge_sort!.

import Atomix

const _RS_BITS  = UInt32(8)
const _RS_SIZE  = UInt32(256)   # 2^_RS_BITS
const _RS_CHUNK = 32            # chunked-scatter chunk width (smaller = cheaper rank; 32 best measured)


# ─── Make any supported scalar type sortable as an unsigned integer ───────────

@inline _to_sort_key(x::UInt32) = x
@inline _to_sort_key(x::UInt64) = x
@inline _to_sort_key(x::Int32)  = reinterpret(UInt32, x) ⊻ 0x80000000
@inline _to_sort_key(x::Int64)  = reinterpret(UInt64, x) ⊻ 0x8000000000000000

@inline function _to_sort_key(x::Float32)
    u = reinterpret(UInt32, x)
    mask = ((u >> 31) * 0xFFFFFFFF) | 0x80000000
    u ⊻ mask
end

@inline function _to_sort_key(x::Float64)
    u = reinterpret(UInt64, x)
    mask = ((u >> 63) * 0xFFFFFFFFFFFFFFFF) | 0x8000000000000000
    u ⊻ mask
end

@inline _rs_digit(x, shift::UInt32, rev::Bool) =
    ((rev ? ~_to_sort_key(x) : _to_sort_key(x)) >> shift) & (_RS_SIZE - 0x1)


# ─── Phase 1: per-pass histogram — generic scan (all backends) ───────────────
# hist[k * num_blocks + b] = count of elements with digit k in block b.
#
# Each thread loads its element's digit into s_digit, then scans s_digit for each
# of its assigned buckets (bucket t, t+NI, t+2*NI, …).  Uses no shared-memory
# atomics, so it is the portable fallback that runs on every backend.

# The trailing Val is always Val(1) on this path; accepted only so the driver can
# call every histogram/scatter kernel with one uniform signature.
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val,
)
    @uniform NI = Int(@groupsize()[1])
    s_digit = @localmem UInt32 (NI,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    # 0xffffffff doesn't match any valid bucket (0–255); OOB elements are neutral.
    i = iblock * NI + ithread
    s_digit[ithread + 1] = UInt32(i < len ? _rs_digit(v[i + 1], shift, rev) : 0xffffffff)
    @synchronize()

    # Thread t handles buckets t, t+NI, t+2*NI, … (covers all 256 when NI ≤ 256).
    bucket = ithread
    while bucket < Int(_RS_SIZE)
        cnt = UInt32(0)
        for jj in 1:NI
            cnt += UInt32(s_digit[jj] == UInt32(bucket))
        end
        hist[bucket * num_blocks + iblock + 1] = cnt
        bucket += NI
    end
end


# ─── Phase 1b: histogram — shared-atomic (backends with shared-mem atomics) ──
# Same per-block 256-bin output, but O(1)/element via a shared-memory atomic
# increment instead of the O(256)/element per-bucket scan above.  Each thread
# processes ITEMS elements (block-strided, so global reads stay coalesced) and a
# block covers block_size*ITEMS elements — shrinking the per-(digit, block)
# histogram, and the exclusive scan over it, by ITEMS×.  Selected by the driver
# only where the backend reports atomics support.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist_atomic!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val{ITEMS},
) where ITEMS
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)
    base = iblock * NI * ITEMS

    j = ithread
    while j < Int(_RS_SIZE)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        i = base + ithread + m * NI
        if i < len
            d = Int(_rs_digit(v[i + 1], shift, rev))
            Atomix.@atomic s_hist[d + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    bucket = ithread
    while bucket < Int(_RS_SIZE)
        hist[bucket * num_blocks + iblock + 1] = s_hist[bucket + 1]
        bucket += NI
    end
end


# ─── Phase 3: scatter — broadcast-read rank (O(N) per thread) ───────────────
# `hist` is already the exclusive-prefix-summed per-block offsets;
# `hist[k * num_blocks + b]` = global start for bucket k, block b (1-indexed).

# The trailing Val is always Val(1) on this path; accepted only so the driver can
# call every histogram/scatter kernel with one uniform signature.
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_gbase = @localmem UInt32       (256,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ 256

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v_in[i + 1]
    end
    j = ithread
    while j < 256
        s_gbase[j + 1] = hist[j * num_blocks + iblock + 1]
        j += NI
    end
    @synchronize()

    my_digit = UInt32(i < len ? _rs_digit(s_elem[ithread + 1], shift, rev) : 0)
    s_digit[ithread + 1] = my_digit
    @synchronize()

    if i < len
        cnt = UInt32(0)
        for jj in UInt32(1):UInt32(ithread)
            cnt += UInt32(s_digit[jj] == my_digit)
        end
        gpos = Int(s_gbase[my_digit + 1]) + Int(cnt)
        v_out[gpos + 1] = s_elem[ithread + 1]
    end
end


# ─── Phase 3b: scatter — chunked stable rank (O(chunk) per thread) ───────────
# The broadcast scatter's rank is O(block_size)/element.  Split the tile into
# 32-wide chunks: per-chunk digit counts (built with shared-memory atomics —
# order doesn't matter for counts) give each chunk's stable base via a cross-chunk
# exclusive prefix, and each element only scans its own chunk (≤32) for the
# intra-chunk part.  Rank = cross-chunk-base[digit] + intra-chunk same-digit
# count — still fully stable, but O(32) instead of O(block_size).
#
# Each thread processes ITEMS tile positions (block-strided, coalesced loads);
# the tile covers block_size*ITEMS elements and must match the histogram's
# block coverage so the per-(digit, block) offsets line up.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter_chunked!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val{ITEMS},
) where ITEMS
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * ITEMS
    @uniform NCH  = (Int(@groupsize()[1]) * ITEMS) ÷ _RS_CHUNK   # number of chunks
    s_elem  = @localmem eltype(v_in) (TILE,)
    s_digit = @localmem UInt32       (TILE,)
    s_gbase = @localmem UInt32       (256,)
    s_chist = @localmem UInt32       (256 * NCH,)      # per-chunk digit counts → bases

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ 256
    base = iblock * TILE

    # Load the tile: keys + digits, in tile-position order (== input order, which
    # the stable rank below relies on).  0xffffffff marks out-of-range positions;
    # it matches no valid digit, so they count and scatter nothing.
    m = 0
    while m < ITEMS
        p = ithread + m * NI
        i = base + p
        if i < len
            k = v_in[i + 1]
            s_elem[p + 1]  = k
            s_digit[p + 1] = _rs_digit(k, shift, rev)
        else
            s_digit[p + 1] = 0xffffffff
        end
        m += 1
    end
    j = ithread
    while j < 256
        s_gbase[j + 1] = hist[j * num_blocks + iblock + 1]
        j += NI
    end
    j = ithread
    while j < 256 * NCH
        s_chist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            Atomix.@atomic s_chist[(p ÷ _RS_CHUNK) * 256 + Int(d) + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    # cross-chunk exclusive prefix per digit (thread d owns digit d, d+NI, …)
    d = ithread
    while d < 256
        acc = UInt32(0)
        for c in 0:NCH-1
            cnt = s_chist[c * 256 + d + 1]
            s_chist[c * 256 + d + 1] = acc
            acc += cnt
        end
        d += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            chunk_start = (p ÷ _RS_CHUNK) * _RS_CHUNK
            cnt = UInt32(0)
            q = chunk_start
            while q < p
                cnt += UInt32(s_digit[q + 1] == d)
                q += 1
            end
            rank = s_chist[(p ÷ _RS_CHUNK) * 256 + Int(d) + 1] + cnt
            gpos = Int(s_gbase[Int(d) + 1]) + Int(rank)
            v_out[gpos + 1] = s_elem[p + 1]
        end
        m += 1
    end
end


# ─── Implementation ──────────────────────────────────────────────────────────

_rs_supported(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64


# Return (min_sort_key, max_sort_key) as UInt64, accounting for descending order.
# Used to detect passes where all elements share the same byte-digit (trivial pass).
#
# Single fused reduction: map each element to its (order-preserving) unsigned sort
# key and reduce to (min_key, max_key) in one pass, instead of two separate full
# reductions over the array (minimum + maximum).  ~halves the range-finding cost.
function _rs_key_range(v::AbstractArray{T}, descending::Bool) where T
    K = typeof(_to_sort_key(zero(T)))   # UInt32 for 32-bit types, UInt64 for 64-bit
    ident = (typemax(K), typemin(K))   # identity for (min, max) over keys
    min_k, max_k = mapreduce(
        x -> (k = _to_sort_key(x); (k, k)),
        (a, b) -> (min(a[1], b[1]), max(a[2], b[2])),
        v;
        init=ident,
        neutral=ident,
    )
    if descending
        # rev=true flips all bits: digit = (~key >> shift) & 0xFF
        # Maximum value → minimum key after negation; swap accordingly.
        UInt64(~max_k), UInt64(~min_k)
    else
        UInt64(min_k), UInt64(max_k)
    end
end


"""
    _radix_sort!(v, backend; lt, by, rev, order, block_size, temp)

In-place GPU LSD radix sort (8-bit, 256 buckets per pass).  Supported types:
`UInt32`, `Int32`, `Float32`, `UInt64`, `Int64`, `Float64`.  Falls back to
[`merge_sort!`](@ref) for any other type or when `lt`/`by` are non-default.
"""
function _radix_sort!(
    v::AbstractArray{T}, backend::Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,
    block_size::Int=256,
    items_per_thread::Int=2,
    temp::Union{Nothing, AbstractArray}=nothing,
) where T

    if !_rs_supported(T) || lt !== isless || by !== identity
        return merge_sort!(v, backend; lt, by, rev, order, block_size, temp)
    end

    n = length(v)
    n <= 1 && return v

    @argcheck ispow2(block_size) && block_size >= 1
    @argcheck items_per_thread >= 1

    descending = (rev === true) || order === Base.Order.Reverse

    # Processing several items per thread shrinks the per-(digit, block) histogram
    # (and its exclusive scan) by the same factor.  Only the fast atomic kernels
    # support it; the portable scan/broadcast fallbacks stay at one item per
    # thread.  The default of 2 keeps the chunked scatter's shared memory within
    # every backend's budget (Metal caps threadgroup memory at 32 KiB); discrete
    # NVIDIA/AMD GPUs are measurably faster still at 4.
    has_atomics = KernelAbstractions.supports_atomics(backend)
    use_fast    = has_atomics && block_size % _RS_CHUNK == 0
    items       = use_fast ? items_per_thread : 1

    num_blocks = cld(n, block_size * items)
    n_passes   = sizeof(T) * 8 ÷ Int(_RS_BITS)   # 4 for 32-bit, 8 for 64-bit

    # Single histogram buffer; no need to zero before each pass — _radix_hist!
    # zero-initializes its own shared-memory histogram and writes directly here.
    hist = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)

    # Reusable scratch for accumulate!'s per-block prefixes (ScanPrefixes uses a
    # 256-thread, 2-elems-per-thread grid → 512 elements per block), so the
    # exclusive prefix sum does not re-allocate on every pass.
    acc_temp = similar(v, UInt32, cld(length(hist), 512))

    p1 = v
    p2 = if !isnothing(temp)
        @argcheck length(temp) >= n && eltype(temp) === T
        temp
    else
        similar(v)
    end

    ndrange = (block_size * num_blocks,)

    # Compute (min, max) sort-key to detect passes where all elements share the
    # same digit → skip the full hist+scan+scatter for that byte position.
    min_key, max_key = _rs_key_range(p1, descending)

    # The fast histogram (O(1)/element atomic counting) and the chunked scatter
    # (per-chunk sub-histograms) both use shared-memory atomics; select them
    # where the backend reports atomics support and fall back to the portable
    # scan/broadcast kernels otherwise.  The chunked scatter additionally needs
    # block_size to be a multiple of its 32-wide chunk.  Both fast kernels must
    # agree on the tile size (block_size*items), so they share the same Val.
    vitems = Val(items)
    hist_kern! = has_atomics ?
        _radix_hist_atomic!(backend, block_size) :
        _radix_hist!(backend, block_size)
    scat_kern! = use_fast ?
        _radix_scatter_chunked!(backend, block_size) :
        _radix_scatter!(backend, block_size)

    n_actual = 0

    for pass in 0:n_passes - 1
        shift = UInt64(pass) * UInt64(_RS_BITS)

        # Trivial pass: all elements share the same byte k AND all higher bytes
        # are also identical (so no element can have a different byte k).
        # Sufficient condition: (min_key >> shift) == (max_key >> shift).
        # Checking just the single byte is WRONG when higher bytes differ.
        (min_key >> shift) == (max_key >> shift) && continue

        shift32 = UInt32(shift)
        # The three kernels run in order on a single backend stream, so no host
        # synchronization is needed between them.
        hist_kern!(hist, p1, shift32, descending, vitems; ndrange)
        accumulate!(+, hist, backend; init=UInt32(0), inclusive=false, temp=acc_temp)
        scat_kern!(p2, p1, hist, shift32, descending, vitems; ndrange)

        p1, p2 = p2, p1
        n_actual += 1
    end

    # p1 holds the result; copy back only if it's in the temp buffer.
    if isodd(n_actual)
        copyto!(v, p1)
    end

    # Block once so the sort is complete on return; the passes only enqueue work.
    KernelAbstractions.synchronize(backend)

    v
end
