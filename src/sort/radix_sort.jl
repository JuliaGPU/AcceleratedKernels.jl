# LSD radix sort (8-bit, 256 buckets).  Stable, GPU-only.
#
# Algorithm per pass (P = 4 passes for 32-bit, 8 for 64-bit):
#   1. _radix_hist!   — each block counts its contiguous chunk into
#                       hist[digit * num_blocks + block]; all 256 threads run
#                       in parallel, each owning one bucket and counting via a
#                       broadcast-read loop over precomputed shared-mem digits.
#   2. accumulate!    — exclusive prefix-sum over the flat hist array gives
#                       scatter offsets:  hist[k*B+b] = # elements with
#                       digit < k (all blocks) + # elements with digit k in
#                       blocks 0..b-1
#   3. _radix_scatter! — each thread stores its digit into shared mem; then
#                        counts preceding same-digit entries via a broadcast
#                        read loop (stable, no atomics); all threads scatter
#                        in parallel using hist[k*B+b] + intra-block rank.
#
# Supported element types: UInt32/64, Int32/64, Float32/64.
# For custom lt/by -> falls back to merge sort.

const _RS_BITS = UInt32(8)
const _RS_SIZE = UInt32(256)   # 2^_RS_BITS


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


# ─── Phase 1: block-level histogram (parallel, one thread per bucket) ─────────
# Elements are staged in s_elem then each element's digit is precomputed once
# into s_digit (avoids calling _to_sort_key repeatedly — for Float64 it involves
# a 64-bit multiply).  Each thread then tallies only its own bucket by scanning
# s_digit: all threads read the same s_digit[jj] each iteration → hardware
# broadcast with zero bank conflicts, no atomics, one global write per thread.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist!(
    hist, @Const(v), shift::UInt32, rev::Bool,
)
    @uniform N  = @groupsize()[1]
    @uniform NI = Int(@groupsize()[1])
    s_elem  = @localmem eltype(v) (N,)
    s_digit = @localmem UInt32   (N,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ 256

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v[i + 1]
    end
    @synchronize()

    block_len = min(NI, len - iblock * NI)
    if ithread < block_len
        s_digit[ithread + 1] = _rs_digit(s_elem[ithread + 1], shift, rev)
    end
    @synchronize()

    j = ithread
    while j < 256
        my_bucket = UInt32(j)
        count = UInt32(0)
        for jj in 1:block_len
            count += UInt32(s_digit[jj] == my_bucket)
        end
        hist[j * num_blocks + iblock + 1] = count
        j += NI
    end
end


# ─── Phase 3: stable scatter (parallel rank via broadcast read) ───────────────
# Each thread stores its digit in s_digit and reads all 256 block offsets from
# the prefix-summed hist into s_gbase — both before the single @synchronize.
# After the barrier each thread counts preceding elements with the same digit via
# a broadcast-read loop (all threads read the same s_digit[jj] each iteration →
# hardware broadcast, zero bank conflicts).  Scatter is then one shared-mem read
# (s_gbase) plus one global write (v_out).

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool,
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

    # Coissue the v_in load and the strided hist reads; both hit global memory
    # simultaneously while the GPU pipelines them.
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


# ─── Implementation ──────────────────────────────────────────────────────────

_rs_supported(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64


"""
    _radix_sort!(
        v::AbstractArray, backend::Backend=get_backend(v);
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )

Sort `v` in-place using a GPU LSD radix sort (8-bit, 256 buckets per pass).

Supported element types: `UInt32`, `Int32`, `Float32`, `UInt64`, `Int64`, `Float64`.
Falls back to merge sort for any other type or when `lt`/`by` are provided.

The temporary buffer `temp` (same type and size as `v`) can be passed to avoid
allocating internally.  `block_size` must be a power of 2.
"""
function _radix_sort!(
    v::AbstractArray{T}, backend::Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
) where T

    # Fall back for unsupported types or custom comparators
    if !_rs_supported(T) || lt !== isless || by !== identity
        return merge_sort!(v, backend; lt, by, rev, order, block_size, temp)
    end

    n = length(v)
    n <= 1 && return v

    @argcheck ispow2(block_size) && block_size >= 1

    descending = (rev === true) || order === Base.Order.Reverse

    num_blocks = cld(n, block_size)
    # hist[k * num_blocks + b] = count of elements with digit k in block b
    hist = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)

    p1 = v
    p2 = if !isnothing(temp)
        @argcheck length(temp) >= n && eltype(temp) === T
        temp
    else
        similar(v)
    end

    n_passes   = sizeof(T) * 8 ÷ Int(_RS_BITS)   # 4 for 32-bit, 8 for 64-bit
    hist_kern! = _radix_hist!(backend, block_size)
    scat_kern! = _radix_scatter!(backend, block_size)
    ndrange    = (block_size * num_blocks,)

    for pass in 0:n_passes - 1
        shift = UInt32(pass) * _RS_BITS

        fill!(hist, 0x0)
        hist_kern!(hist, p1, shift, descending; ndrange)
        KernelAbstractions.synchronize(backend)

        accumulate!(+, hist, backend; init=UInt32(0), inclusive=false)
        KernelAbstractions.synchronize(backend)

        scat_kern!(p2, p1, hist, shift, descending; ndrange)
        KernelAbstractions.synchronize(backend)

        p1, p2 = p2, p1
    end

    # After an even number of passes p1 === v (result already in v).
    # After an odd number, p1 is the temp buffer; copy back.
    if isodd(n_passes)
        copyto!(v, p1)
    end

    v
end
