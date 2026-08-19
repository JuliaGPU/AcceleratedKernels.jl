# Stable GPU LSD radix sort with 8-bit digits.
# The atomic kernels use several items per thread; scan kernels are the portable
# fallback for backends without shared-memory atomics.

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
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val,
)
    @uniform NI = Int(@groupsize()[1])
    s_digit = @localmem UInt32 (NI,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    i = iblock * NI + ithread
    s_digit[ithread + 1] = UInt32(i < len ? _rs_digit(v[i + 1], shift, rev) : 0xffffffff)
    @synchronize()

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


# Histogram with shared-memory atomics.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist_atomic!(
    hist, @Const(v), shift::UInt32, rev::Bool, ::Val{ITEMS},
) where ITEMS
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)
    j = ithread
    while j < Int(_RS_SIZE)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    m = 0
    while m < ITEMS
        i = iblock * NI * ITEMS + ithread + m * NI
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


# Stable scatter without atomics.
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_gbase = @localmem UInt32       (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v_in[i + 1]
    end
    j = ithread
    while j < Int(_RS_SIZE)
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


# Stable scatter with chunked ranks.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter_chunked!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool, ::Val{ITEMS},
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
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)
    m = 0
    while m < ITEMS
        p = ithread + m * NI
        i = iblock * TILE + p
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
    while j < Int(_RS_SIZE)
        s_gbase[j + 1] = hist[j * num_blocks + iblock + 1]
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
            v_out[gpos + 1] = s_elem[p + 1]
        end
        m += 1
    end
end



# Single-block sort for small arrays.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_sort_block!(
    v, rev::Bool, ::Val{NPASS},
) where NPASS
    @uniform NI   = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * 2
    @uniform NCH  = (Int(@groupsize()[1]) * 2) ÷ _RS_CHUNK
    s_a     = @localmem eltype(v) (TILE,)
    s_b     = @localmem eltype(v) (TILE,)
    s_digit = @localmem UInt32    (TILE,)
    s_chist = @localmem UInt32    (Int(_RS_SIZE) * NCH,)
    s_loff  = @localmem UInt32    (Int(_RS_SIZE),)

    it = Int(@index(Local, Linear)) - 1
    n  = Int(length(v))

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

const _RS_LOCAL_MEMORY_LIMIT = 32 * 1024

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


# Return the extrema of the transformed sort keys.
function _rs_key_range(v::AbstractArray{T}, descending::Bool) where T
    K = typeof(_to_sort_key(zero(T)))
    ident = (typemax(K), typemin(K))
    min_k, max_k = mapreduce(
        x -> (k = _to_sort_key(x); (k, k)),
        (a, b) -> (min(a[1], b[1]), max(a[2], b[2])),
        v;
        init=ident,
        neutral=ident,
    )
    if descending
        UInt64(~max_k), UInt64(~min_k)
    else
        UInt64(min_k), UInt64(max_k)
    end
end


"""
    _radix_sort!(v, backend; descending, block_size, temp)

In-place GPU radix sort for supported 32- and 64-bit integers and floats.
"""
function _radix_sort!(
    v::AbstractArray{T}, backend::Backend=get_backend(v);
    descending::Bool=false,
    block_size::Int=256,
    items_per_thread::Int=2,
    temp::Union{Nothing, AbstractArray}=nothing,
) where T
    n = length(v)
    n <= 1 && return v

    @argcheck ispow2(block_size) && block_size >= 1
    @argcheck items_per_thread >= 1
    @argcheck _rs_portable_local_memory(T, block_size) <= _RS_LOCAL_MEMORY_LIMIT

    has_atomics = KernelAbstractions.supports_atomics(backend)
    use_fast    = has_atomics && block_size % _RS_CHUNK == 0 &&
                  _rs_fast_local_memory(T, block_size, items_per_thread) <= _RS_LOCAL_MEMORY_LIMIT
    items       = use_fast ? items_per_thread : 1

    n_passes = sizeof(T) * 8 ÷ Int(_RS_BITS)

    if use_fast && _rs_block_local_memory(T, block_size) <= _RS_LOCAL_MEMORY_LIMIT &&
       n <= 2 * block_size
        _radix_sort_block!(backend, block_size)(
            v, descending, Val(n_passes); ndrange=block_size)
        KernelAbstractions.synchronize(backend)
        return v
    end

    num_blocks = cld(n, block_size * items)

    hist = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)

    acc_temp = similar(v, UInt32, cld(length(hist), 512))

    p1 = v
    p2 = if !isnothing(temp)
        @argcheck length(temp) >= n && eltype(temp) === T
        temp
    else
        similar(v)
    end

    ndrange = (block_size * num_blocks,)

    min_key, max_key = _rs_key_range(p1, descending)

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
        hist_kern!(hist, p1, shift32, descending, vitems; ndrange)
        accumulate!(+, hist, backend; init=UInt32(0), inclusive=false, temp=acc_temp)
        scat_kern!(p2, p1, hist, shift32, descending, vitems; ndrange)

        p1, p2 = p2, p1
        n_actual += 1
    end

    if isodd(n_actual)
        copyto!(v, p1)
    end

    KernelAbstractions.synchronize(backend)

    v
end
