# Stable GPU LSD radix sort with 8-bit digits.
# The atomic kernels use several items per thread; scan kernels are the portable
# fallback for backends without shared-memory atomics.
#
# The kernels sort the slices of a slices.jl layout, a whole-array sort being one flat slice. Each
# slice is split into `blocks_per_slice` blocks, and the digit histograms are laid out
# slice-major: `hist[(slice * 256 + digit) * blocks_per_slice + block]`. An exclusive scan of all
# of them then gives every (slice, digit, block) its position in the concatenation of the sorted
# slices, from which the scatter subtracts the slice's start. The subtraction wraps, so positions
# within a slice are right as long as the slice fits in a UInt32.

import Atomix

const RADIX_BITS = UInt32(8)
const RADIX_DIGITS = UInt32(256)
# Positions are UInt32s, so this many elements at most are sorted together
const RADIX_MAX_LEN = Int64(2)^32


# Sort keys

@inline uint_map(x::UInt32) = x
@inline uint_map(x::UInt64) = x
@inline uint_map(x::Int32)  = reinterpret(UInt32, x) ⊻ 0x80000000
@inline uint_map(x::Int64)  = reinterpret(UInt64, x) ⊻ 0x8000000000000000

@inline function uint_map(x::Float32)
    u = reinterpret(UInt32, x)
    mask = ((u >> 31) * 0xFFFFFFFF) | 0x80000000
    ifelse(isnan(x), typemax(UInt32), u ⊻ mask)
end

@inline function uint_map(x::Float64)
    u = reinterpret(UInt64, x)
    mask = ((u >> 63) * 0xFFFFFFFFFFFFFFFF) | 0x8000000000000000
    ifelse(isnan(x), typemax(UInt64), u ⊻ mask)
end

@inline radix_digit(x, shift::UInt32, rev::Bool) =
    ((rev ? ~uint_map(x) : uint_map(x)) >> shift) & (RADIX_DIGITS - 0x1)


# Histogram without atomics.
@kernel inbounds=true cpu=false unsafe_indices=true function radix_histogram_kernel!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val, layout, blocks_per_slice::Int,
)
    @uniform NI = Int(@groupsize()[1])
    s_digit = @localmem UInt32 (NI,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, blocks_per_slice)
    src = slice(v, layout, islice)
    len = layout.len

    i = iblk * NI + ithread
    s_digit[ithread + 1] = UInt32(i < len ? radix_digit(src[i + 1], shift, rev) : 0xffffffff)
    @synchronize()

    base = islice * Int(RADIX_DIGITS) * blocks_per_slice + iblk
    bucket = ithread
    while bucket < Int(RADIX_DIGITS)
        cnt = UInt32(0)
        for jj in 1:NI
            cnt += UInt32(s_digit[jj] == UInt32(bucket))
        end
        hist[base + bucket * blocks_per_slice + 1] = cnt
        bucket += NI
    end
end


# Histogram with shared-memory atomics.

@kernel inbounds=true cpu=false unsafe_indices=true function radix_histogram_atomic_kernel!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val{ITEMS}, layout, blocks_per_slice::Int,
) where ITEMS
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(RADIX_DIGITS),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, blocks_per_slice)
    src = slice(v, layout, islice)
    len = layout.len
    j = ithread
    while j < Int(RADIX_DIGITS)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        i = iblk * NI * ITEMS + ithread + m * NI
        if i < len
            d = Int(radix_digit(src[i + 1], shift, rev))
            Atomix.@atomic s_hist[d + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    base = islice * Int(RADIX_DIGITS) * blocks_per_slice + iblk
    bucket = ithread
    while bucket < Int(RADIX_DIGITS)
        hist[base + bucket * blocks_per_slice + 1] = s_hist[bucket + 1]
        bucket += NI
    end
end


# Stable scatter without atomics.
@kernel inbounds=true cpu=false unsafe_indices=true function radix_scatter_kernel!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val, layout,
    blocks_per_slice::Int,
)
    @uniform NI  = Int(@groupsize()[1])
    s_elem         = @localmem eltype(v_in) (NI,)
    s_digit        = @localmem UInt32       (NI,)
    s_digit_offset = @localmem UInt32       (Int(RADIX_DIGITS),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, blocks_per_slice)
    src = slice(v_in, layout, islice)
    dst = slice(v_out, layout, islice)
    len = layout.len

    i = iblk * NI + ithread
    if i < len
        s_elem[ithread + 1] = src[i + 1]
    end
    base = islice * Int(RADIX_DIGITS) * blocks_per_slice + iblk
    start = (islice * len) % UInt32
    j = ithread
    while j < Int(RADIX_DIGITS)
        s_digit_offset[j + 1] = hist[base + j * blocks_per_slice + 1] - start
        j += NI
    end
    @synchronize()

    my_digit = UInt32(i < len ? radix_digit(s_elem[ithread + 1], shift, rev) : 0)
    s_digit[ithread + 1] = my_digit
    @synchronize()

    if i < len
        cnt = UInt32(0)
        for jj in UInt32(1):UInt32(ithread)
            cnt += UInt32(s_digit[jj] == my_digit)
        end
        gpos = Int(s_digit_offset[my_digit + 1]) + Int(cnt)
        dst[gpos + 1] = s_elem[ithread + 1]
    end
end


# Stable scatter with chunked ranks.

@kernel inbounds=true cpu=false unsafe_indices=true function radix_scatter_chunked_kernel!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val{ITEMS}, ::Val{CH},
    layout, blocks_per_slice::Int,
) where {ITEMS, CH}
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * ITEMS
    @uniform NCH  = (Int(@groupsize()[1]) * ITEMS) ÷ CH
    s_elem         = @localmem eltype(v_in) (TILE,)
    s_digit        = @localmem UInt32       (TILE,)
    s_digit_offset = @localmem UInt32       (Int(RADIX_DIGITS),)
    s_chunk_counts = @localmem UInt32       (Int(RADIX_DIGITS) * NCH,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, blocks_per_slice)
    src = slice(v_in, layout, islice)
    dst = slice(v_out, layout, islice)
    len = layout.len
    m = 0
    while m < ITEMS
        p = ithread + m * NI
        i = iblk * TILE + p
        if i < len
            k = src[i + 1]
            s_elem[p + 1]  = k
            s_digit[p + 1] = radix_digit(k, shift, rev)
        else
            s_digit[p + 1] = 0xffffffff
        end
        m += 1
    end
    base = islice * Int(RADIX_DIGITS) * blocks_per_slice + iblk
    start = (islice * len) % UInt32
    j = ithread
    while j < Int(RADIX_DIGITS)
        s_digit_offset[j + 1] = hist[base + j * blocks_per_slice + 1] - start
        j += NI
    end
    j = ithread
    while j < Int(RADIX_DIGITS) * NCH
        s_chunk_counts[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            Atomix.@atomic s_chunk_counts[(p ÷ CH) * Int(RADIX_DIGITS) + Int(d) + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    d = ithread
    while d < Int(RADIX_DIGITS)
        acc = UInt32(0)
        for c in 0:NCH-1
            cnt = s_chunk_counts[c * Int(RADIX_DIGITS) + d + 1]
            s_chunk_counts[c * Int(RADIX_DIGITS) + d + 1] = acc
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
            chunk_start = (p ÷ CH) * CH
            # Fixed-trip form avoids a POCL LLVM loop-vectorizer failure.
            cnt = UInt32(0)
            for r in 0:CH - 1
                q = chunk_start + r
                cnt += UInt32((q < p) & (s_digit[q + 1] == d))
            end
            rank = s_chunk_counts[(p ÷ CH) * Int(RADIX_DIGITS) + Int(d) + 1] + cnt
            gpos = Int(s_digit_offset[Int(d) + 1]) + Int(rank)
            dst[gpos + 1] = s_elem[p + 1]
        end
        m += 1
    end
end



# Sorts of slices that fit one block, one block per slice, two elements per thread: the slices
# are at most `2 * block_size` long, as for the merge sort's tiles.

@kernel inbounds=true cpu=false unsafe_indices=true function radix_sort_block_kernel!(
    data, rev::Bool, ::Val{NPASS}, ::Val{CH}, layout,
) where {NPASS, CH}
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * 2
    @uniform NCH  = (Int(@groupsize()[1]) * 2) ÷ CH
    s_a            = @localmem eltype(data) (TILE,)
    s_b            = @localmem eltype(data) (TILE,)
    s_digit        = @localmem UInt32       (TILE,)
    s_chunk_counts = @localmem UInt32       (Int(RADIX_DIGITS) * NCH,)
    s_digit_offset = @localmem UInt32       (Int(RADIX_DIGITS),)

    it = Int(@index(Local, Linear)) - 1
    v  = slice(data, layout, Int(@index(Group, Linear)) - 1)
    n  = layout.len

    m = 0
    while m < 2
        p = it + m * NI
        if p < n
            s_a[p + 1] = v[p + 1]
        end
        m += 1
    end
    @synchronize()

    pass = 0
    while pass < NPASS
        sh = UInt32(pass) * RADIX_BITS
        src = iseven(pass) ? s_a : s_b
        dst = iseven(pass) ? s_b : s_a

        j = it
        while j < Int(RADIX_DIGITS) * NCH
            s_chunk_counts[j + 1] = UInt32(0)
            j += NI
        end
        j = it
        while j < Int(RADIX_DIGITS)
            s_digit_offset[j + 1] = UInt32(0)
            j += NI
        end
        @synchronize()

        m = 0
        while m < 2
            p = it + m * NI
            if p < n
                d = radix_digit(src[p + 1], sh, rev)
                s_digit[p + 1] = d
                Atomix.@atomic s_chunk_counts[(p ÷ CH) * Int(RADIX_DIGITS) + Int(d) + 1] +=
                    UInt32(1)
            end
            m += 1
        end
        @synchronize()

        d = it
        while d < Int(RADIX_DIGITS)
            acc = UInt32(0)
            c = 0
            while c < NCH
                cnt = s_chunk_counts[c * Int(RADIX_DIGITS) + d + 1]
                s_chunk_counts[c * Int(RADIX_DIGITS) + d + 1] = acc
                acc += cnt
                c += 1
            end
            s_digit_offset[d + 1] = acc
            d += NI
        end
        @synchronize()

        if it == 0
            run = UInt32(0)
            dd = 0
            while dd < Int(RADIX_DIGITS)
                t = s_digit_offset[dd + 1]
                s_digit_offset[dd + 1] = run
                run += t
                dd += 1
            end
        end
        @synchronize()

        m = 0
        while m < 2
            p = it + m * NI
            if p < n
                d = s_digit[p + 1]
                chunk_start = (p ÷ CH) * CH
                # (fixed-trip, as in the chunked scatter)
                cnt = UInt32(0)
                for r in 0:CH - 1
                    q = chunk_start + r
                    cnt += UInt32((q < p) & (s_digit[q + 1] == d))
                end
                rank = s_chunk_counts[(p ÷ CH) * Int(RADIX_DIGITS) + Int(d) + 1] + cnt
                dst[Int(s_digit_offset[Int(d) + 1]) + Int(rank) + 1] = src[p + 1]
            end
            m += 1
        end
        @synchronize()

        pass += 1
    end

    res = iseven(NPASS) ? s_a : s_b
    m = 0
    while m < 2
        p = it + m * NI
        if p < n
            v[p + 1] = res[p + 1]
        end
        m += 1
    end
end


# Driver

radix_supports_eltype(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64

@inline function radix_portable_localmem(::Type{T}, block_size::Int) where T
    block_size * (sizeof(T) + sizeof(UInt32)) + Int(RADIX_DIGITS) * sizeof(UInt32)
end

@inline function radix_chunked_localmem(::Type{T}, block_size::Int, items::Int, chunk::Int) where T
    tile = block_size * items
    chunks = tile ÷ chunk
    tile * (sizeof(T) + sizeof(UInt32)) + Int(RADIX_DIGITS) * sizeof(UInt32) * (chunks + 1)
end

@inline function radix_block_localmem(::Type{T}, block_size::Int, chunk::Int) where T
    tile = 2 * block_size
    chunks = tile ÷ chunk
    2 * tile * sizeof(T) + tile * sizeof(UInt32) + Int(RADIX_DIGITS) * sizeof(UInt32) * (chunks + 1)
end


# The extrema of the transformed sort keys, as a reduction over `(key, key)` pairs
struct RadixKeyPair end
@inline (::RadixKeyPair)(x) = (k = uint_map(x); (k, k))
struct RadixKeyExtrema end
@inline (::RadixKeyExtrema)(a, b) = (min(a[1], b[1]), max(a[2], b[2]))

function radix_key_range_setup(v::AbstractArray{T}, backend::Backend) where T
    K = typeof(uint_map(zero(T)))
    ident = (typemax(K), typemin(K))
    return mapreduce_setup(RadixKeyPair(), RadixKeyExtrema(), v, backend, ident, ident, nothing,
                           :, Auto()), ident
end

# Return the extrema of the transformed sort keys, with the scratch of `radix_key_range_setup`
function radix_key_range(v::AbstractArray{T}, backend::Backend, rev::Bool, bufs) where T
    s, ident = radix_key_range_setup(v, backend)
    min_k, max_k = mapreduce_run(RadixKeyPair(), RadixKeyExtrema(), v, s, ident, bufs)
    if rev
        UInt64(~max_k), UInt64(~min_k)
    else
        UInt64(min_k), UInt64(max_k)
    end
end


# The launch configuration of `radix_sort!` for slices of length `len`: whether it uses the
# chunked kernels (which need the chunks to tile the block), the items per thread they get, and
# whether one block sorts each slice in local memory
function radix_config(::Type{T}, len, backend, block_size, items_per_thread, chunk_size) where T
    has_atomics = KernelAbstractions.supports_atomics(backend)
    chunked     = has_atomics && block_size % chunk_size == 0 &&
                  radix_chunked_localmem(T, block_size, items_per_thread, chunk_size) <=
                  LOCAL_MEMORY_BUDGET
    items       = chunked ? items_per_thread : 1
    one_block   = chunked &&
                  radix_block_localmem(T, block_size, chunk_size) <= LOCAL_MEMORY_BUDGET &&
                  len <= 2 * block_size
    return (; has_atomics, chunked, items, one_block)
end

# The number of blocks of the radix passes. Tiny slices make it exceed the number of elements, so
# it and the sizes derived from it are checked: on 32-bit hosts, they could overflow.
radix_blocks(layout, tile) = Base.checked_mul(slice_count(layout), cld(layout.len, tile))

# The plan of `radix_sort!`: the output of every other pass, the digit histograms of every
# block, and the plans of the histograms' scan and the key range reduction
function radix_plan(a, v::AbstractArray{T}, backend, dims) where T
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return Plan(backend, a, (;))
    c = radix_config(T, layout.len, backend, a.block_size, a.items_per_thread, a.chunk_size)
    c.one_block && return Plan(backend, a, (;))
    hist_len = Base.checked_mul(Int(RADIX_DIGITS), radix_blocks(layout, a.block_size * c.items))
    scan = scan_setup(+, UInt32, UInt32, (hist_len,), backend; init=UInt32(0),
                      inclusive=false).plan
    key_range = first(radix_key_range_setup(v, backend)).plan
    return Plan(backend, a, (; temp=scratch(T, size(v)), hist=scratch(UInt32, hist_len), scan,
                             key_range))
end


# GPU radix sort of `v` (or of each slice along `dims`), in place, with the scratch buffers of
# `radix_plan`
function radix_sort!(
    v::AbstractArray{T}, backend::Backend, bufs::NamedTuple;
    rev::Bool, block_size::Int, items_per_thread::Int, chunk_size::Int, dims::Union{Colon, Integer},
) where T
    layout = slice_layout(v, dims)
    len = layout.len
    (isempty(v) || len <= 1) && return v

    (; has_atomics, chunked, items, one_block) =
        radix_config(T, len, backend, block_size, items_per_thread, chunk_size)

    n_passes = sizeof(T) * 8 ÷ Int(RADIX_BITS)

    if one_block
        radix_sort_block_kernel!(backend, block_size)(
            v, rev, Val(n_passes), Val(chunk_size), layout;
            ndrange=Base.checked_mul(block_size, slice_count(layout)))
        return v
    end

    blocks_per_slice = cld(len, block_size * items)
    num_blocks = radix_blocks(layout, block_size * items)
    hist = bufs.hist
    p1 = v
    p2 = bufs.temp

    ndrange = (Base.checked_mul(block_size, num_blocks),)

    min_key, max_key = radix_key_range(p1, backend, rev, bufs.key_range)

    histogram! = has_atomics ?
        radix_histogram_atomic_kernel!(backend, block_size) :
        radix_histogram_kernel!(backend, block_size)
    scatter! = chunked ?
        radix_scatter_chunked_kernel!(backend, block_size) :
        radix_scatter_kernel!(backend, block_size)
    scatter_args = chunked ? (Val(items), Val(chunk_size)) : (Val(items),)

    passes_run = 0

    for pass in 0:n_passes - 1
        shift = UInt64(pass) * UInt64(RADIX_BITS)

        # The keys agree on this digit and all higher ones, so the remaining passes are no-ops
        (min_key >> shift) == (max_key >> shift) && break

        shift32 = UInt32(shift)
        histogram!(hist, p1, shift32, rev, Val(items), layout, blocks_per_slice; ndrange)
        accumulate_nested!(+, hist, bufs.scan; backend, init=UInt32(0), inclusive=false)
        scatter!(p2, p1, hist, shift32, rev, scatter_args..., layout, blocks_per_slice; ndrange)

        p1, p2 = p2, p1
        passes_run += 1
    end

    if isodd(passes_run)
        copyto_on!(v, p1, backend)
    end

    v
end
