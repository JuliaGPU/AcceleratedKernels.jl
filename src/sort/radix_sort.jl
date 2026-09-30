# Stable GPU LSD radix sort with 8-bit digits.
# The atomic kernels use several items per thread; scan kernels are the portable
# fallback for backends without shared-memory atomics.
#
# The kernels sort the slices of a slices.jl layout, a whole-array sort being one flat slice. Each
# slice is split into `bps` blocks, and the digit histograms are laid out slice-major:
# `hist[(slice * 256 + digit) * bps + block]`. An exclusive scan of all of them then gives every
# (slice, digit, block) its position in the concatenation of the sorted slices, from which the
# scatter subtracts the slice's start. The subtraction wraps, so positions within a slice are
# right as long as the slice fits in a UInt32.

import Atomix

const _RS_BITS = UInt32(8)
const _RS_SIZE = UInt32(256)
const _RS_CHUNK = 32


# Sort keys

@inline _to_sort_key(x::UInt32) = x
@inline _to_sort_key(x::UInt64) = x
@inline _to_sort_key(x::Int32)  = reinterpret(UInt32, x) ⊻ 0x80000000
@inline _to_sort_key(x::Int64)  = reinterpret(UInt64, x) ⊻ 0x8000000000000000

@inline function _to_sort_key(x::Float32)
    u = reinterpret(UInt32, x)
    mask = ((u >> 31) * 0xFFFFFFFF) | 0x80000000
    ifelse(isnan(x), typemax(UInt32), u ⊻ mask)
end

@inline function _to_sort_key(x::Float64)
    u = reinterpret(UInt64, x)
    mask = ((u >> 63) * 0xFFFFFFFFFFFFFFFF) | 0x8000000000000000
    ifelse(isnan(x), typemax(UInt64), u ⊻ mask)
end

@inline _rs_digit(x, shift::UInt32, rev::Bool) =
    ((rev ? ~_to_sort_key(x) : _to_sort_key(x)) >> shift) & (_RS_SIZE - 0x1)


# Histogram without atomics.
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val, layout, bps::Int,
)
    @uniform NI = Int(@groupsize()[1])
    s_digit = @localmem UInt32 (NI,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, bps)
    src = slice(v, layout, islice)
    len = layout.len

    i = iblk * NI + ithread
    s_digit[ithread + 1] = UInt32(i < len ? _rs_digit(src[i + 1], shift, rev) : 0xffffffff)
    @synchronize()

    base = islice * Int(_RS_SIZE) * bps + iblk
    bucket = ithread
    while bucket < Int(_RS_SIZE)
        cnt = UInt32(0)
        for jj in 1:NI
            cnt += UInt32(s_digit[jj] == UInt32(bucket))
        end
        hist[base + bucket * bps + 1] = cnt
        bucket += NI
    end
end


# Histogram with shared-memory atomics.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist_atomic!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val{ITEMS}, layout, bps::Int,
) where ITEMS
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, bps)
    src = slice(v, layout, islice)
    len = layout.len
    j = ithread
    while j < Int(_RS_SIZE)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        i = iblk * NI * ITEMS + ithread + m * NI
        if i < len
            d = Int(_rs_digit(src[i + 1], shift, rev))
            Atomix.@atomic s_hist[d + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    base = islice * Int(_RS_SIZE) * bps + iblk
    bucket = ithread
    while bucket < Int(_RS_SIZE)
        hist[base + bucket * bps + 1] = s_hist[bucket + 1]
        bucket += NI
    end
end


# Stable scatter without atomics.
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val, layout, bps::Int,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_gbase = @localmem UInt32       (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, bps)
    src = slice(v_in, layout, islice)
    dst = slice(v_out, layout, islice)
    len = layout.len

    i = iblk * NI + ithread
    if i < len
        s_elem[ithread + 1] = src[i + 1]
    end
    base = islice * Int(_RS_SIZE) * bps + iblk
    start = (islice * len) % UInt32
    j = ithread
    while j < Int(_RS_SIZE)
        s_gbase[j + 1] = hist[base + j * bps + 1] - start
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
        dst[gpos + 1] = s_elem[ithread + 1]
    end
end


# Stable scatter with chunked ranks.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter_chunked!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val{ITEMS}, layout, bps::Int,
) where ITEMS
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * ITEMS
    @uniform NCH  = (Int(@groupsize()[1]) * ITEMS) ÷ _RS_CHUNK
    s_elem  = @localmem eltype(v_in) (TILE,)
    s_digit = @localmem UInt32       (TILE,)
    s_gbase = @localmem UInt32       (Int(_RS_SIZE),)
    s_chist = @localmem UInt32       (Int(_RS_SIZE) * NCH,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(layout, iblock, bps)
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
            s_digit[p + 1] = _rs_digit(k, shift, rev)
        else
            s_digit[p + 1] = 0xffffffff
        end
        m += 1
    end
    base = islice * Int(_RS_SIZE) * bps + iblk
    start = (islice * len) % UInt32
    j = ithread
    while j < Int(_RS_SIZE)
        s_gbase[j + 1] = hist[base + j * bps + 1] - start
        j += NI
    end
    j = ithread
    while j < Int(_RS_SIZE) * NCH
        s_chist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        p = ithread + m * NI
        d = s_digit[p + 1]
        if d != 0xffffffff
            Atomix.@atomic s_chist[(p ÷ _RS_CHUNK) * Int(_RS_SIZE) + Int(d) + 1] += UInt32(1)
        end
        m += 1
    end
    @synchronize()

    d = ithread
    while d < Int(_RS_SIZE)
        acc = UInt32(0)
        for c in 0:NCH-1
            cnt = s_chist[c * Int(_RS_SIZE) + d + 1]
            s_chist[c * Int(_RS_SIZE) + d + 1] = acc
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
            # Fixed-trip form avoids a POCL LLVM loop-vectorizer failure.
            cnt = UInt32(0)
            for r in 0:_RS_CHUNK - 1
                q = chunk_start + r
                cnt += UInt32((q < p) & (s_digit[q + 1] == d))
            end
            rank = s_chist[(p ÷ _RS_CHUNK) * Int(_RS_SIZE) + Int(d) + 1] + cnt
            gpos = Int(s_gbase[Int(d) + 1]) + Int(rank)
            dst[gpos + 1] = s_elem[p + 1]
        end
        m += 1
    end
end



# Sorts of slices that fit one block, one block per slice.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_sort_block!(
    data, rev::Bool, ::Val{NPASS}, layout,
) where NPASS
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * 2
    @uniform NCH  = (Int(@groupsize()[1]) * 2) ÷ _RS_CHUNK
    s_a     = @localmem eltype(data) (TILE,)
    s_b     = @localmem eltype(data) (TILE,)
    s_digit = @localmem UInt32    (TILE,)
    s_chist = @localmem UInt32    (Int(_RS_SIZE) * NCH,)
    s_loff  = @localmem UInt32    (Int(_RS_SIZE),)

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
        sh = UInt32(pass) * _RS_BITS
        src = iseven(pass) ? s_a : s_b
        dst = iseven(pass) ? s_b : s_a

        j = it
        while j < Int(_RS_SIZE) * NCH
            s_chist[j + 1] = UInt32(0)
            j += NI
        end
        j = it
        while j < Int(_RS_SIZE)
            s_loff[j + 1] = UInt32(0)
            j += NI
        end
        @synchronize()

        m = 0
        while m < 2
            p = it + m * NI
            if p < n
                d = _rs_digit(src[p + 1], sh, rev)
                s_digit[p + 1] = d
                Atomix.@atomic s_chist[(p ÷ _RS_CHUNK) * Int(_RS_SIZE) + Int(d) + 1] += UInt32(1)
            end
            m += 1
        end
        @synchronize()

        d = it
        while d < Int(_RS_SIZE)
            acc = UInt32(0)
            c = 0
            while c < NCH
                cnt = s_chist[c * Int(_RS_SIZE) + d + 1]
                s_chist[c * Int(_RS_SIZE) + d + 1] = acc
                acc += cnt
                c += 1
            end
            s_loff[d + 1] = acc
            d += NI
        end
        @synchronize()

        if it == 0
            run = UInt32(0)
            dd = 0
            while dd < Int(_RS_SIZE)
                t = s_loff[dd + 1]
                s_loff[dd + 1] = run
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
                chunk_start = (p ÷ _RS_CHUNK) * _RS_CHUNK
                cnt = UInt32(0)
                for r in 0:_RS_CHUNK - 1
                    q = chunk_start + r
                    cnt += UInt32((q < p) & (s_digit[q + 1] == d))
                end
                rank = s_chist[(p ÷ _RS_CHUNK) * Int(_RS_SIZE) + Int(d) + 1] + cnt
                dst[Int(s_loff[Int(d) + 1]) + Int(rank) + 1] = src[p + 1]
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

_rs_supported(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64

@inline function _rs_portable_local_memory(::Type{T}, block_size::Int) where T
    block_size * (sizeof(T) + sizeof(UInt32)) + Int(_RS_SIZE) * sizeof(UInt32)
end

@inline function _rs_fast_local_memory(::Type{T}, block_size::Int, items::Int) where T
    tile = block_size * items
    chunks = tile ÷ _RS_CHUNK
    tile * (sizeof(T) + sizeof(UInt32)) + Int(_RS_SIZE) * sizeof(UInt32) * (chunks + 1)
end

@inline function _rs_block_local_memory(::Type{T}, block_size::Int) where T
    tile = 2 * block_size
    chunks = tile ÷ _RS_CHUNK
    2 * tile * sizeof(T) + tile * sizeof(UInt32) + Int(_RS_SIZE) * sizeof(UInt32) * (chunks + 1)
end


# The extrema of the transformed sort keys, as a reduction over `(key, key)` pairs
struct _RSKeyPair end
@inline (::_RSKeyPair)(x) = (k = _to_sort_key(x); (k, k))
struct _RSMinMax end
@inline (::_RSMinMax)(a, b) = (min(a[1], b[1]), max(a[2], b[2]))

function _rs_key_range_setup(v::AbstractArray{T}, backend::Backend) where T
    K = typeof(_to_sort_key(zero(T)))
    ident = (typemax(K), typemin(K))
    return _mapreduce_setup(_RSKeyPair(), _RSMinMax(), v, backend, ident, ident, nothing, :,
                            Auto()), ident
end

# Return the extrema of the transformed sort keys, with the scratch of `_rs_key_range_setup`
function _rs_key_range(v::AbstractArray{T}, backend::Backend, descending::Bool, bufs) where T
    s, ident = _rs_key_range_setup(v, backend)
    min_k, max_k = _mapreduce_run(_RSKeyPair(), _RSMinMax(), v, s, ident, bufs)
    if descending
        UInt64(~max_k), UInt64(~min_k)
    else
        UInt64(min_k), UInt64(max_k)
    end
end


# The launch configuration of `_radix_sort!` for slices of length `len`: whether it uses the
# chunked kernels, the items per thread they get, and whether one block sorts each slice in local
# memory
function _rs_config(::Type{T}, len, backend, block_size, items_per_thread) where T
    has_atomics = KernelAbstractions.supports_atomics(backend)
    use_fast    = has_atomics && block_size % _RS_CHUNK == 0 &&
                  _rs_fast_local_memory(T, block_size, items_per_thread) <= LOCAL_MEMORY_BUDGET
    items       = use_fast ? items_per_thread : 1
    one_block   = use_fast && _rs_block_local_memory(T, block_size) <= LOCAL_MEMORY_BUDGET &&
                  len <= 2 * block_size
    return (; has_atomics, use_fast, items, one_block)
end

# The number of blocks of the radix passes. Tiny slices make it exceed the number of elements, so
# it and the sizes derived from it are checked: on 32-bit hosts, they could overflow.
_rs_num_blocks(layout, tile) = Base.checked_mul(slice_count(layout), cld(layout.len, tile))

# The scratch of `_radix_sort!`, and the algorithms of the operations it calls: the output of
# every other pass, the digit histograms of every block, and the histograms' scan and the key
# range reduction
function _radix_plan(a, v::AbstractArray{T}, backend, dims) where T
    layout = slice_layout(v, dims)
    (isempty(v) || layout.len <= 1) && return (;), (;)
    c = _rs_config(T, layout.len, backend, a.block_size, a.items_per_thread)
    c.one_block && return (;), (;)
    hist_len = Base.checked_mul(Int(_RS_SIZE), _rs_num_blocks(layout, a.block_size * c.items))
    scan = _accumulate_setup(+, UInt32, UInt32, (hist_len,), backend; init=UInt32(0),
                             inclusive=false).plan
    key_range = first(_rs_key_range_setup(v, backend)).plan
    return (; temp=_buffer(T, size(v)), hist=_buffer(UInt32, hist_len), scan=scan.sizes,
            key_range=key_range.sizes), (; scan=scan.alg, key_range=key_range.alg)
end


"""
    _radix_sort!(v, backend, bufs; descending, block_size, items_per_thread, dims)

In-place GPU radix sort of `v`, or of its slices along `dims`, for supported 32- and 64-bit
integers and floats, with the scratch buffers of `_radix_plan`.
"""
function _radix_sort!(
    v::AbstractArray{T}, backend::Backend, bufs::NamedTuple;
    descending::Bool=false,
    block_size::Int=256,
    items_per_thread::Int=2,
    dims::Union{Colon, Integer}=Colon(),
) where T
    layout = slice_layout(v, dims)
    len = layout.len
    (isempty(v) || len <= 1) && return v

    @argcheck ispow2(block_size) && block_size >= 1
    @argcheck items_per_thread >= 1
    @argcheck _rs_portable_local_memory(T, block_size) <= LOCAL_MEMORY_BUDGET

    (; has_atomics, use_fast, items, one_block) = _rs_config(T, len, backend, block_size, items_per_thread)

    n_passes = sizeof(T) * 8 ÷ Int(_RS_BITS)

    if one_block
        _radix_sort_block!(backend, block_size)(
            v, descending, Val(n_passes), layout;
            ndrange=Base.checked_mul(block_size, slice_count(layout)))
        KernelAbstractions.synchronize(backend)
        return v
    end

    bps = cld(len, block_size * items)
    num_blocks = _rs_num_blocks(layout, block_size * items)
    hist = bufs.hist
    p1 = v
    p2 = bufs.temp

    ndrange = (Base.checked_mul(block_size, num_blocks),)

    min_key, max_key = _rs_key_range(p1, backend, descending, bufs.key_range)

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

        (min_key >> shift) == (max_key >> shift) && continue

        shift32 = UInt32(shift)
        hist_kern!(hist, p1, shift32, descending, vitems, layout, bps; ndrange)
        _accumulate_nested!(+, hist, bufs.scan; backend, init=UInt32(0), inclusive=false)
        scat_kern!(p2, p1, hist, shift32, descending, vitems, layout, bps; ndrange)

        p1, p2 = p2, p1
        n_actual += 1
    end

    if isodd(n_actual)
        copyto!(v, p1)
    end

    KernelAbstractions.synchronize(backend)

    v
end
