# Portable GPU bitonic sort on KernelAbstractions primitives. A sorting network has no
# data-dependent branching, so it maps well onto GPUs; inputs that fit a single workgroup sort
# entirely in shared memory. Non-power-of-two lengths are padded with a sentinel (`typemax`
# ascending, `typemin` descending) so it sorts to the tail.

# Radix's eltypes: comparable, with a `typemax`/`typemin` sentinel.
_bs_supported(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64

# Single-workgroup shared-memory budget in bytes; overridable per backend. Matches the radix sort.
_bs_shmem_bytes(::Backend) = 32 * 1024
@inline _prevpow2(x::Int) = 1 << (8 * sizeof(Int) - leading_zeros(x) - 1)


# Sort one CAP-sized window from scratch in shared memory; the window's global base fixes each
# comparator's direction. Used for a whole small array and to build the windows of the large path.
@kernel cpu=false inbounds=true function _bitonic_full!(
    w, descending::Bool, ::Val{CAP}, ::Val{BS}, ::Val{IPT},
) where {CAP, BS, IPT}
    tile = @localmem eltype(w) (CAP,)
    blk  = @index(Group, Linear)
    t    = Int(@index(Local, Linear)) - 1
    base = Int(blk - 0x1) * CAP

    m = 0
    while m < IPT
        pos = t + m * BS
        @inbounds tile[pos + 1] = w[base + pos + 1]
        m += 1
    end
    @synchronize()

    kklog = 1
    while (1 << kklog) <= CAP
        jlog = kklog - 1
        while jlog >= 0
            j = 1 << jlog
            p = t
            while p < (CAP >> 1)
                i = ((p >> jlog) << (jlog + 1)) | (p & (j - 1))
                partner = i + j
                asc = ((base + i) & (1 << kklog)) == 0
                a = @inbounds tile[i + 1]
                b = @inbounds tile[partner + 1]
                if (a > b) == (asc != descending)
                    @inbounds tile[i + 1] = b
                    @inbounds tile[partner + 1] = a
                end
                p += BS
            end
            @synchronize()
            jlog -= 1
        end
        kklog += 1
    end

    m = 0
    while m < IPT
        pos = t + m * BS
        @inbounds w[base + pos + 1] = tile[pos + 1]
        m += 1
    end
end


# Finish the short strides (`j_start` down to 0, all < CAP) of merge level `kklog` for one window
# in shared memory; every comparator's pair stays inside the window.
@kernel cpu=false inbounds=true function _bitonic_block!(
    w, kklog::Int, j_start::Int, descending::Bool,
    ::Val{CAP}, ::Val{BS}, ::Val{IPT},
) where {CAP, BS, IPT}
    tile = @localmem eltype(w) (CAP,)
    blk  = @index(Group, Linear)
    t    = Int(@index(Local, Linear)) - 1
    base = Int(blk - 0x1) * CAP
    kk = 1 << kklog

    m = 0
    while m < IPT
        pos = t + m * BS
        @inbounds tile[pos + 1] = w[base + pos + 1]
        m += 1
    end
    @synchronize()

    jlog = j_start
    while jlog >= 0
        j = 1 << jlog
        p = t
        while p < (CAP >> 1)
            i = ((p >> jlog) << (jlog + 1)) | (p & (j - 1))
            partner = i + j
            asc = ((base + i) & kk) == 0
            a = @inbounds tile[i + 1]
            b = @inbounds tile[partner + 1]
            if (a > b) == (asc != descending)
                @inbounds tile[i + 1] = b
                @inbounds tile[partner + 1] = a
            end
            p += BS
        end
        @synchronize()
        jlog -= 1
    end

    m = 0
    while m < IPT
        pos = t + m * BS
        @inbounds w[base + pos + 1] = tile[pos + 1]
        m += 1
    end
end


# One global compare-exchange for a large-stride merge level, launched one thread per pair
# (`npow ÷ 2`): every lane works, no divergent guard, and consecutive threads stay coalesced.
@kernel cpu=false inbounds=true function _bitonic_global!(
    w, kklog::Int, jlog::Int, descending::Bool,
)
    p = Int(@index(Global, Linear)) - 1
    j = 1 << jlog
    i = ((p >> jlog) << (jlog + 1)) | (p & (j - 1))
    partner = i + j
    asc = (i & (1 << kklog)) == 0
    a = @inbounds w[i + 1]
    b = @inbounds w[partner + 1]
    if (a > b) == (asc != descending)
        @inbounds w[i + 1] = b
        @inbounds w[partner + 1] = a
    end
end


function _bitonic_sort!(
    v::AbstractVector{T}, backend::Backend;
    descending::Bool,
    block_size::Union{Nothing, Int}=nothing,
) where T
    len = length(v)
    len <= 1 && return v

    npow = nextpow(2, len)
    budget = _prevpow2(_bs_shmem_bytes(backend) ÷ sizeof(T))
    cap = min(npow, budget)
    bs = isnothing(block_size) ? min(256, cap) : min(_prevpow2(block_size), cap)
    ipt = cap ÷ bs

    # Pad to a power of two; the sentinel sinks to the tail so real values fill 1:len.
    pad = descending ? typemin(T) : typemax(T)
    if npow == len
        w = v
    else
        w = similar(v, npow)
        copyto!(w, 1, v, 1, len)
        fill!(view(w, (len + 1):npow), pad)
    end

    if npow <= cap
        _bitonic_full!(backend, bs)(w, descending, Val(cap), Val(bs), Val(ipt); ndrange = bs)
    else
        nlog = trailing_zeros(npow)
        caplog = trailing_zeros(cap)
        nblocks = npow ÷ cap
        full = _bitonic_full!(backend, bs)
        block = _bitonic_block!(backend, bs)
        global_step = _bitonic_global!(backend, 256)

        # Build sorted windows, then merge them: global passes for j >= cap, one shared-memory
        # batch for the remaining j < cap of each merge level.
        full(w, descending, Val(cap), Val(bs), Val(ipt); ndrange = nblocks * bs)
        for kklog in (caplog + 1):nlog
            jlog = kklog - 1
            while jlog >= caplog
                global_step(w, kklog, jlog, descending; ndrange = npow ÷ 2)
                jlog -= 1
            end
            block(w, kklog, caplog - 1, descending, Val(cap), Val(bs), Val(ipt);
                  ndrange = nblocks * bs)
        end
    end
    KernelAbstractions.synchronize(backend)

    npow == len || copyto!(v, 1, w, 1, len)
    v
end
