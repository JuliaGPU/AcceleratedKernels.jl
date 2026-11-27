# Selection: the elements at given positions of the sorted array (or of each sorted slice along
# `dims`), without sorting it.
#
# `RadixSelect` is an MSD radix select over the order-preserving keys of radix_sort.jl, with 8-bit
# digits, following one or two ranks at a time. Every pass histograms the current digit of the
# elements that agree with a rank's prefix of digits resolved so far, and the bucket holding the
# rank extends that prefix. Three kernel families cover the slice lengths:
#
# - `select_count_kernel!`, for slices of up to `count_max_len` elements: each thread counts the
#   elements of its slice that sort before its own (equal ones by position), which gives its
#   element's rank; several slices share a block.
# - `select_block_kernel!`, for slices of up to `block_max_len` elements: one block per slice, its
#   keys in registers, the passes in local memory, stopping once every rank has one candidate.
# - the global passes (`select_hist_kernel!` and `select_resolve_kernel!`), for longer slices:
#   several blocks per slice histogram a digit into global memory, then one block per slice
#   resolves it. The state lives on the device, so that the passes need no host round trips. The
#   first pass also finds the extrema of the keys, whose common leading digits need no passes,
#   and once a rank's candidates fit a buffer of `1/COMPACT_FRACTION` of the slice, the next pass
#   copies them there, for the remaining passes to read instead of the whole slice.
#
# `SortSelect` sorts a copy and gathers the ranks: every element type, ordering and number of
# ranks.


"""
    SelectAlgorithm <: Algorithm

Supertype of the selection algorithms: [`RadixSelect`](@ref) and [`SortSelect`](@ref).
"""
abstract type SelectAlgorithm <: Algorithm end

"""
    RadixSelect(; block_size=nothing, items_per_thread=nothing, blocks=nothing)

GPU radix select, for `UInt32`, `Int32`, `Float32`, `UInt64`, `Int64` and `Float64` (floats
ordered like `isless`) and one or two ranks. Slices of up to 8192 elements are selected by one
block each, in local memory; longer ones by passes over the slices, with `blocks` blocks in all of
`block_size` threads (a power of two from 256 to 1024) and `items_per_thread` elements each.
"""
Base.@kwdef struct RadixSelect <: SelectAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
    blocks::Union{Nothing, Int} = nothing
end

"""
    SortSelect(; sort_alg=Auto(stable=false))

Select by sorting a copy of the input with `sort_alg` (any [sorting algorithm](@ref
SortAlgorithm), or `Auto`), then gathering the requested ranks: any element type that `isless`
orders, and any number of ranks.
"""
Base.@kwdef struct SortSelect{A <: Algorithm} <: SelectAlgorithm
    sort_alg::A = Auto(stable=false)
end


"""
    SelectTuning(; kwargs...)

Values that pick `RadixSelect`'s kernels and fill its unset fields on one device, as returned by
[`select_tuning`](@ref).

- `count_max_len`, `block_max_len`: the longest slices `RadixSelect` selects by counting and in
  one block.
- `block_size`, `items_per_thread`, `blocks`: the settings of the global passes.
"""
Base.@kwdef struct SelectTuning
    # (on an RTX 5080, counting slices of 128 Float32s takes 2.4 ms for 2^19 of them, the
    # one-block radix select 3.5 ms; for slices of 192, 3.4 against 2.5 ms)
    count_max_len::Int = 160
    block_max_len::Int = 8192
    # (more items per thread take more registers, and fewer blocks fit: on an RTX 5080, 4 is 25%
    # faster than 8)
    block_size::Int = 512
    items_per_thread::Int = 4
    blocks::Int = 2048
end
"""
    select_tuning(backend, T) -> SelectTuning

The selection tuning for element type `T` on `backend`'s current device. AK defines this generic
method; AK's package extensions add one method per backend type.
"""
select_tuning(::Backend, ::Type) = SelectTuning()
tuning_type(::typeof(select_tuning)) = SelectTuning

# Candidates are compacted once they fit in this fraction of a slice
const COMPACT_FRACTION = 8


# Keys

@inline select_key(x, rev::Bool) = rev ? ~uint_map(x) : uint_map(x)
select_key_type(::Type{T}) where {T} = typeof(uint_map(zero(T)))

@inline key_unmap(::Type{UInt32}, k::UInt32) = k
@inline key_unmap(::Type{UInt64}, k::UInt64) = k
@inline key_unmap(::Type{Int32}, k::UInt32) = reinterpret(Int32, k ⊻ 0x80000000)
@inline key_unmap(::Type{Int64}, k::UInt64) = reinterpret(Int64, k ⊻ 0x8000000000000000)
# (a NaN's key, `typemax`, maps back to a NaN)
@inline key_unmap(::Type{Float32}, k::UInt32) =
    reinterpret(Float32, k >> 31 == 0x1 ? k ⊻ 0x80000000 : ~k)
@inline key_unmap(::Type{Float64}, k::UInt64) =
    reinterpret(Float64, k >> 63 == 0x1 ? k ⊻ 0x8000000000000000 : ~k)
@inline select_value(::Type{T}, k, rev::Bool) where {T} = key_unmap(T, rev ? ~k : k)

# The `j`-th of one or two ranks, without indexing the tuple dynamically
@inline rank_at(ranks::Tuple, j) = UInt32(j == 1 ? first(ranks) : last(ranks))

# `f(state, item, m)` folded over the items `m = 1, 2, ...` of a thread's tuple, unrolled (a
# recursion over the tail loses inference of the state for 8 or more items)
@generated function fold_items(f, st, ks::NTuple{N, Any}) where {N}
    steps = [:(st = f(st, ks[$m], $m)) for m in 1:N]
    return quote
        Base.@_inline_meta
        $(steps...)
        st
    end
end

# The bits of the digits before pass `pass`, and the shift of its digit
@inline pass_shift(::Type{K}, pass) where {K} = 8 * sizeof(K) - 8 * (pass + 1)
@inline pass_mask(::Type{K}, pass) where {K} =
    pass == 0 ? zero(K) : ~zero(K) << (8 * sizeof(K) - 8 * pass)

# Find the buckets of the ranks `r1` and `r2` (the second for `NS == 2`) among the candidates with
# prefixes `p1` and `p2`, from the histograms `hist[h0 + 1:h0 + 256]` (and the next 256 bins for
# the second rank, unless both prefixes agree and share the first histogram), and write the
# extended prefixes, the ranks within the buckets and the buckets' counts. All threads of the block
# must call it; it synchronizes at the end.
@inline function resolve_buckets!(@context, s_prefix, s_rank, s_cnt, s_scan, hist, h0, p1::K, p2::K,
                                  r1, r2, shift, t, ::Val{NS}, ::Val{W}) where {K, NS, W}
    B = @groupsize()[1]
    shared = p1 == p2
    nscan = NS == 2 && !shared ? 2 : 1
    s = 1
    while s <= nscan
        c = t < 256 ? hist[h0 + (s - 1) * 256 + t + 1] : UInt32(0)
        excl, _ = block_exclusive_scan(@context, +, s_scan, c, UInt32(0), B, t, Val(W))
        if t < 256
            if s == 1 && excl < r1 <= excl + c
                s_prefix[1] = p1 | (K(t) << shift)
                s_rank[1] = r1 - excl
                s_cnt[1] = c
            end
            if NS == 2 && s == nscan && excl < r2 <= excl + c
                s_prefix[2] = p2 | (K(t) << shift)
                s_rank[2] = r2 - excl
                s_cnt[2] = c
            end
        end
        @synchronize()
        s += 1
    end
end


# Counting select of short slices, `B ÷ SPAN` of them per block of `B` threads

@kernel inbounds=true cpu=false unsafe_indices=true function select_count_kernel!(
    dst, @Const(src), rev::Bool, ranks::NTuple{NS, Int}, src_layout, dst_layout, ::Val{SPAN},
) where {NS, SPAN}
    @uniform B = Int(@groupsize()[1])
    @uniform T = eltype(src)
    @uniform K = select_key_type(eltype(src))
    s_key = @localmem K (B,)

    t = Int(@index(Local, Linear)) - 1
    islice = (Int(@index(Group, Linear)) - 1) * (B ÷ SPAN) + t ÷ SPAN
    pos = t % SPAN
    len = src_layout.len
    valid = pos < len && islice < slice_count(src_layout)
    x = valid ? slice(src, src_layout, islice)[pos + 1] : zero(T)
    k = select_key(x, rev)
    s_key[t + 1] = k
    @synchronize()

    if valid
        base = t - pos
        less = UInt32(0)
        eq = UInt32(0)
        q = 0
        while q < len
            kq = s_key[base + q + 1]
            less += UInt32(kq < k)
            eq += UInt32(kq == k)
            q += 1
        end
        d = slice(dst, dst_layout, islice)
        j = 1
        while j <= NS
            r = rank_at(ranks, j)
            if less < r <= less + eq
                # (equal keys rank by their positions, so that one thread writes each rank)
                before = less
                if eq > 0x1
                    q = 0
                    while q < pos
                        before += UInt32(s_key[base + q + 1] == k)
                        q += 1
                    end
                end
                if r == before + 0x1
                    d[j] = x
                end
            end
            j += 1
        end
    end
end


# Radix select of one slice per block, `ITEMS` keys per thread in registers

@kernel inbounds=true cpu=false unsafe_indices=true function select_block_kernel!(
    dst, @Const(src), rev::Bool, ranks::NTuple{NS, Int}, src_layout, dst_layout, ::Val{ITEMS},
    ::Val{W},
) where {NS, ITEMS, W}
    @uniform B = Int(@groupsize()[1])
    @uniform T = eltype(src)
    @uniform K = select_key_type(eltype(src))
    @uniform P = sizeof(K)
    s_hist = @localmem UInt32 (256 * NS,)
    s_scan = @localmem UInt32 (block_slots(B, Val(W)),)
    s_prefix = @localmem K (NS,)
    s_rank = @localmem UInt32 (NS,)
    s_cnt = @localmem UInt32 (NS,)

    t = Int(@index(Local, Linear)) - 1
    islice = Int(@index(Group, Linear)) - 1
    v = slice(src, src_layout, islice)
    len = src_layout.len
    keys = ntuple(Val(ITEMS)) do m
        @inline
        i = t + (m - 1) * B
        i < len ? select_key(@inbounds(v[i + 1]), rev) : typemax(K)
    end
    if t < NS
        s_prefix[t + 1] = zero(K)
        s_rank[t + 1] = rank_at(ranks, t + 1)
        s_cnt[t + 1] = UInt32(len)
    end
    @synchronize()

    pass = 0
    while pass < P
        shift = pass_shift(K, pass)
        mask = pass_mask(K, pass)
        p1 = s_prefix[1]
        p2 = NS == 2 ? s_prefix[2] : p1
        r1 = s_rank[1]
        r2 = NS == 2 ? s_rank[2] : r1
        shared = p1 == p2
        j = t
        while j < 256 * NS
            s_hist[j + 1] = UInt32(0)
            j += B
        end
        @synchronize()

        # (runs of equal bins among a thread's keys take one atomic, so that equal keys do not
        # serialize on one bin)
        run_bin, run = fold_items((-1, UInt32(0)), keys) do acc, k, m
            @inline
            rb, rn = acc
            t + (m - 1) * B < len || return acc
            digit = Int((k >> shift) & 0xff)
            bin = (k & mask) == p1 ? digit :
                  (!shared && (k & mask) == p2) ? 256 + digit : -1
            bin == rb && return (rb, rn + UInt32(1))
            rb >= 0 && @inbounds Atomix.@atomic :monotonic s_hist[rb + 1] += rn
            return (bin, UInt32(1))
        end
        run_bin >= 0 && Atomix.@atomic :monotonic s_hist[run_bin + 1] += run
        @synchronize()

        resolve_buckets!(@context, s_prefix, s_rank, s_cnt, s_scan, s_hist, 0, p1, p2, r1, r2,
                         shift, t, Val(NS), Val(W))

        pass += 1
        done = s_cnt[1] == 0x1 && (NS == 1 || s_cnt[2] == 0x1)
        done && break
    end

    d = slice(dst, dst_layout, islice)
    if pass == P
        # every digit is resolved: the prefix is the key
        if t < NS
            d[t + 1] = select_value(T, s_prefix[t + 1], rev)
        end
    else
        # one candidate is left per rank, which its thread writes
        fmask = pass_mask(K, pass)
        q1 = s_prefix[1]
        q2 = NS == 2 ? s_prefix[2] : q1
        fold_items(nothing, keys) do _, k, m
            @inline
            if t + (m - 1) * B < len
                (k & fmask) == q1 && @inbounds(d[1] = select_value(T, k, rev))
                NS == 2 && (k & fmask) == q2 && @inbounds(d[2] = select_value(T, k, rev))
            end
            nothing
        end
    end
end


# Global passes. The state of slice `i` (zero-based) is `kst[2i + 1:2i + 2]` (the two ranks'
# prefixes) and `ust[5i + 1:5i + 5]` (the two ranks within their candidates, the compacted count,
# the compaction mode: 0 for none yet, 1 for compacting in the next pass that runs, 2 for
# compacted, and the first pass that runs). `hist` holds two sets of histograms per slice, for
# even and odd passes, each `NS * 256` bins.

const UST = 5
const KST = 2

@kernel inbounds=true cpu=false unsafe_indices=true function select_init_kernel!(
    hist, kst, ust, ranks::NTuple{NS, Int}, len::Int,
) where {NS}
    @uniform B = Int(@groupsize()[1])
    t = Int(@index(Local, Linear)) - 1
    islice = Int(@index(Group, Linear)) - 1
    K = eltype(kst)
    j = t
    while j < 2 * NS * 256
        hist[islice * 2 * NS * 256 + j + 1] = UInt32(0)
        j += B
    end
    if t == 0
        u = islice * UST
        ust[u + 1] = rank_at(ranks, 1)
        ust[u + 2] = rank_at(ranks, 2)
        ust[u + 3] = UInt32(0)
        ust[u + 4] = UInt32(0)
        ust[u + 5] = UInt32(0)
        kk = islice * KST
        kst[kk + 1] = zero(K)
        kst[kk + 2] = zero(K)
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function select_hist_kernel!(
    hist, comp, pmin, pmax, @Const(kst), ust, @Const(src), rev::Bool, src_layout, pass::Int,
    bps::Int, cap::Int, ::Val{NS}, ::Val{ITEMS}, ::Val{W},
) where {NS, ITEMS, W}
    @uniform B = Int(@groupsize()[1])
    @uniform TILE = Int(@groupsize()[1]) * ITEMS
    @uniform K = select_key_type(eltype(src))
    s_hist = @localmem UInt32 (256 * NS,)
    s_stage = @localmem K (TILE,)
    s_count = @localmem UInt32 (2,)
    s_red = @localmem K (block_slots(B, Val(W)),)

    t = Int(@index(Local, Linear)) - 1
    islice, iblk = slice_block(src_layout, Int(@index(Group, Linear)) - 1, bps)
    u = islice * UST
    kk = islice * KST
    if pass >= Int(ust[u + 5])
        mode = ust[u + 4]
        p1 = kst[kk + 1]
        p2 = NS == 2 ? kst[kk + 2] : p1
        shared = p1 == p2
        compacting = mode == 0x1
        shift = pass_shift(K, pass)
        mask = pass_mask(K, pass)
        v = slice(src, src_layout, islice)
        n = mode == 0x2 ? Int(ust[u + 3]) : src_layout.len

        j = t
        while j < 256 * NS
            s_hist[j + 1] = UInt32(0)
            j += B
        end
        if t == 0
            s_count[1] = UInt32(0)
        end
        @synchronize()

        # (each thread loads its keys of a tile before histogramming them, so that the loads
        # overlap)
        st = (-1, UInt32(0), typemax(K), typemin(K))
        tile = iblk
        while tile * TILE < n
            i0 = tile * TILE + t
            keys = ntuple(Val(ITEMS)) do m
                @inline
                i = i0 + (m - 1) * B
                i < n ? (mode == 0x2 ? @inbounds(comp[islice * cap + i + 1]) :
                                       select_key(@inbounds(v[i + 1]), rev)) : zero(K)
            end
            st = fold_items(st, keys) do acc, k, m
                @inline
                rb, rn, kl, kh = acc
                i0 + (m - 1) * B < n || return acc
                if pass == 0
                    kl = min(kl, k)
                    kh = max(kh, k)
                end
                digit = Int((k >> shift) & 0xff)
                bin = (k & mask) == p1 ? digit :
                      (!shared && (k & mask) == p2) ? 256 + digit : -1
                if compacting && bin >= 0
                    slot = (Atomix.@atomic :monotonic s_count[1] += UInt32(1)) - UInt32(1)
                    @inbounds s_stage[slot + 1] = k
                end
                bin == rb && return (rb, rn + UInt32(1), kl, kh)
                rb >= 0 && @inbounds Atomix.@atomic :monotonic s_hist[rb + 1] += rn
                return (bin, UInt32(1), kl, kh)
            end
            if compacting
                @synchronize()
                if t == 0
                    staged = s_count[1]
                    s_count[2] = (Atomix.@atomic :monotonic ust[u + 3] += staged) - staged
                end
                @synchronize()
                staged = Int(s_count[1])
                base = islice * cap + Int(s_count[2])
                j = t
                while j < staged
                    comp[base + j + 1] = s_stage[j + 1]
                    j += B
                end
                @synchronize()
                if t == 0
                    s_count[1] = UInt32(0)
                end
                @synchronize()
            end
            tile += bps
        end
        run_bin, run, kmin, kmax = st
        run_bin >= 0 && Atomix.@atomic :monotonic s_hist[run_bin + 1] += run
        @synchronize()

        h0 = (islice * 2 + pass % 2) * NS * 256
        j = t
        while j < 256 * NS
            c = s_hist[j + 1]
            c != 0x0 && Atomix.@atomic :monotonic hist[h0 + j + 1] += c
            j += B
        end

        if pass == 0
            lo = block_reduce(@context, min, s_red, kmin, t, Val(W))
            @synchronize()
            hi = block_reduce(@context, max, s_red, kmax, t, Val(W))
            if t == 0
                pmin[islice * bps + iblk + 1] = lo
                pmax[islice * bps + iblk + 1] = hi
            end
        end
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function select_resolve_kernel!(
    dst, hist, @Const(pmin), @Const(pmax), kst, ust, dst_layout, ::Val{T}, rev::Bool, pass::Int,
    bps::Int, cap::Int, nout::Int, ::Val{NS}, ::Val{WS}, ::Val{W},
) where {T, NS, WS, W}
    @uniform B = Int(@groupsize()[1])
    @uniform K = select_key_type(T)
    @uniform P = sizeof(K)
    s_scan = @localmem UInt32 (block_slots(B, Val(WS)),)
    s_red = @localmem K (block_slots(B, Val(W)),)
    s_new = @localmem K (NS,)
    s_rank = @localmem UInt32 (NS,)
    s_cnt = @localmem UInt32 (NS,)
    s_ext = @localmem K (2,)

    t = Int(@index(Local, Linear)) - 1
    islice = Int(@index(Group, Linear)) - 1
    u = islice * UST
    kk = islice * KST
    if pass >= Int(ust[u + 5])
        p1 = kst[kk + 1]
        p2 = NS == 2 ? kst[kk + 2] : p1
        r1 = ust[u + 1]
        r2 = ust[u + 2]
        shared = p1 == p2
        shift = pass_shift(K, pass)
        h0 = (islice * 2 + pass % 2) * NS * 256

        if pass == 0
            lo = typemax(K)
            hi = typemin(K)
            j = t
            while j < bps
                lo = min(lo, pmin[islice * bps + j + 1])
                hi = max(hi, pmax[islice * bps + j + 1])
                j += B
            end
            lo = block_reduce(@context, min, s_red, lo, t, Val(W))
            @synchronize()
            hi = block_reduce(@context, max, s_red, hi, t, Val(W))
            if t == 0
                s_ext[1] = lo
                s_ext[2] = hi
            end
            @synchronize()
        end

        resolve_buckets!(@context, s_new, s_rank, s_cnt, s_scan, hist, h0, p1, p2, r1, r2,
                         shift, t, Val(NS), Val(WS))

        if t == 0
            next = pass + 1
            if pass == 0
                # the keys agree on the leading digits of their extrema, whose passes are skipped
                lo, hi = s_ext[1], s_ext[2]
                nd = lo == hi ? P : leading_zeros(lo ⊻ hi) ÷ 8
                if nd > 1
                    common = lo & pass_mask(K, nd)
                    s = 1
                    while s <= NS
                        s_new[s] = common
                        s += 1
                    end
                    next = nd
                end
            end
            s = 1
            while s <= NS
                kst[kk + s] = s_new[s]
                ust[u + s] = s_rank[s]
                s += 1
            end
            ust[u + 5] = UInt32(next)
            total = (NS == 1 || s_new[1] == s_new[NS]) ? Int(s_cnt[1]) :
                    Int(s_cnt[1]) + Int(s_cnt[NS])
            mode = ust[u + 4]
            if mode == 0x1
                ust[u + 4] = UInt32(2)
            elseif mode == 0x0 && total <= cap && next < P
                ust[u + 4] = UInt32(1)
            end
        end
    end

    # (the histograms of the next pass, which the pass before last read)
    h1 = (islice * 2 + (pass + 1) % 2) * NS * 256
    j = t
    while j < NS * 256
        hist[h1 + j + 1] = UInt32(0)
        j += B
    end

    if pass == P - 1 && t == 0
        d = slice(dst, dst_layout, islice)
        s = 1
        while s <= nout
            d[s] = select_value(T, kst[kk + s], rev)
            s += 1
        end
    end
end


# Host API

"""
    select!(
        dst::AbstractArray,
        src::AbstractArray,
        ranks::AbstractVector{<:Integer};
        dims::Union{Colon, Integer}=:,
        rev::Bool=false,
        alg::Algorithm=Auto(),
        backend=nothing,
        workspace=nothing,
    ) -> dst

Write to `dst` the elements of `src` at the sorted positions `ranks`, without sorting `src`:
`dst[j]` is equal under `isless` (reversed with `rev=true`) to `sort(vec(src); rev)[ranks[j]]`.
`ranks` is a host vector of 1-based positions, in any order and possibly repeated.

With `dims=:`, `src` is selected from as one vector, and `dst` is a vector of `length(ranks)`.
With an integer `dims`, every slice of `src` along `dims` is selected from, with the same ranks:
`dst` has the size of `src` with dimension `dims` replaced by `length(ranks)`.

`src` is not modified. `dst` must not overlap it, nor hold any element twice (as a view with
repeated indices would). As
for `isless`, `-0.0` sorts before `0.0` and NaNs after every other value; a selected NaN is a
NaN, though not necessarily with the bits of one in `src`.

`alg` is [`Auto()`](@ref Auto) by default, which picks [`RadixSelect`](@ref) on GPUs for the
element types it supports and one or two ranks, and [`SortSelect`](@ref) otherwise (and always
on the host).

`backend` is derived from `dst` and `src`; pass it only for arrays that do not determine their
backend. `workspace` takes the scratch memory of a [`workspace`](@ref) made for the same call.

# Examples
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 1_000_000))
m = CuArray{Float32}(undef, 2)
AK.select!(m, v, [500_000, 500_001])          # the two middle elements, for a median

A = CuArray(rand(Float32, 1000, 64))
q = CuArray{Float32}(undef, 1, 64)
AK.select!(q, A, [100]; dims=1)                # the 100th smallest element of each column
```
"""
function select!(dst::AbstractArray, src::AbstractArray, ranks::AbstractVector{<:Integer};
                 dims::Union{Colon, Integer}=Colon(), rev::Bool=false, alg::Algorithm=Auto(),
                 backend::Union{Nothing, Backend}=nothing, workspace=nothing)
    s = select_setup(dst, src, ranks; dims, rev, alg, backend)
    select_with!(s.plan, workspace, dst, src, s.ranks, dims, rev)
    return dst
end

# (behind a function barrier: with `Auto()`, the plan is one of several types, which the
# number of ranks, known at run time, decides)
function select_with!(p::Plan, workspace, dst, src, ranks, dims, rev)
    bufs = scratch_buffers(p, workspace, dst, src)
    isempty(dst) && return dst
    select_run!(p.alg, one_based(dst), one_based(src), ranks, p.backend, dims, rev, bufs)
end

plan(::typeof(select!), dst::AbstractArray, src::AbstractArray, ranks::AbstractVector{<:Integer};
     kwargs...) = select_setup(dst, src, ranks; kwargs...).plan

"""
    select(src::AbstractArray, ranks::AbstractVector{<:Integer}; kwargs...)

Out-of-place [`select!`](@ref): allocate the destination for `src`, `ranks` and `dims`, select
into it with the same keywords, and return it.
"""
function select(src::AbstractArray, ranks::AbstractVector{<:Integer};
                dims::Union{Colon, Integer}=Colon(), backend::Union{Nothing, Backend}=nothing,
                kwargs...)
    backend = resolve_backend(backend, src)
    dst = similar_on(backend, src, eltype(src), select_size(src, length(ranks), dims))
    return select!(dst, src, ranks; dims, backend, kwargs...)
end

plan(::typeof(select), src::AbstractArray, ranks::AbstractVector{<:Integer};
     backend=nothing, kwargs...) =
    select_plan(src, ranks; backend=resolve_backend(backend, src), kwargs...).plan

# The size of the destination of `nr` ranks along `dims`
select_size(src, nr, ::Colon) = (nr,)
function select_size(src, nr, dims::Integer)
    1 <= dims <= ndims(src) ||
        throw(ArgumentError("select!: dimension $dims is out of range for $(ndims(src)) dimensions"))
    return ntuple(d -> d == dims ? nr : size(src, d), ndims(src))
end

function select_setup(dst, src, ranks; dims=Colon(), backend=nothing, kwargs...)
    backend = resolve_backend(backend, dst, src)
    T = eltype(src)
    eltype(dst) === T || throw(ArgumentError(
        "select!: the destination's element type $(eltype(dst)) differs from the source's $T"))
    Base.mightalias(dst, src) &&
        throw(ArgumentError("select!: the destination must not overlap the source"))
    nr = length(ranks)
    want = select_size(src, nr, dims)
    dims isa Colon ? dst isa AbstractVector && length(dst) == nr || throw(DimensionMismatch(
                         "select!: with dims=:, the destination must be a vector of length $nr")) :
                     size(dst) == want || throw(DimensionMismatch(
                         "select!: the destination must have size $want, got $(size(dst))"))
    return select_plan(src, ranks; dims, backend, kwargs...)
end

# The plan of selecting `ranks` from `src`, and the ranks as the runs take them
function select_plan(src, ranks; dims=Colon(), rev::Bool=false, alg::Algorithm=Auto(),
                     backend::Backend)
    T = eltype(src)
    nr = length(ranks)
    select_size(src, nr, dims)
    len = dims isa Colon ? length(src) : size(src, dims)
    nslices = dims isa Colon ? 1 : Base.prod(d -> d == dims ? 1 : size(src, d), 1:ndims(src))
    for r in ranks
        1 <= r <= len || throw(ArgumentError(
            "select!: rank $r is out of range for slices of $len elements"))
    end
    rk = Vector{Int}(undef, nr)
    for (j, r) in enumerate(ranks)
        rk[j] = r
    end
    a = resolve_select(alg, backend, T, nr, len)
    sizes = select_scratch(a, src, len, nslices, nr, dims, rev, backend)
    # (an empty result is written by no kernel)
    if nr == 0 || nslices == 0
        sizes = no_elements(sizes)
    end
    return (; plan=Plan(backend, a, sizes), ranks=rk)
end

function resolve_select(alg::Algorithm, backend::Backend, ::Type{T}, nr, len) where {T}
    t = tuning(select_tuning, backend, T)
    a = alg isa Auto ? select_algorithm(backend, t, T, nr, len) : alg
    a = fill_defaults(a, t)
    check_select(a, backend, t, T, nr, len)
    return a
end

select_algorithm(backend, t::SelectTuning, ::Type{T}, nr, len) where {T} =
    !runs_threads(backend) && radix_supports_eltype(T) && nr <= 2 && len < typemax(UInt32) ?
    RadixSelect() : SortSelect()

fill_defaults(a::RadixSelect, t::SelectTuning) =
    RadixSelect(something(a.block_size, t.block_size),
                something(a.items_per_thread, t.items_per_thread), something(a.blocks, t.blocks))
fill_defaults(a::SortSelect, t::SelectTuning) = a
fill_defaults(a::Algorithm, t::SelectTuning) =
    throw(ArgumentError("$(algname(a)) is not a selection algorithm"))

function check_select(a::RadixSelect, backend, t::SelectTuning, ::Type{T}, nr, len) where {T}
    require_kernels(a, backend)
    radix_supports_eltype(T) || throw(ArgumentError(
        "RadixSelect does not support element type $T; it supports 32- and 64-bit integers " *
        "and floats"))
    nr <= 2 || throw(ArgumentError("RadixSelect selects at most two ranks, got $nr"))
    len < typemax(UInt32) ||
        throw(ArgumentError("RadixSelect selects from at most 2^32 - 2 elements per slice"))
    a.block_size >= 256 && ispow2(a.block_size) && a.block_size <= 1024 || throw(ArgumentError(
        "RadixSelect: `block_size` must be a power of two from 256 to 1024"))
    a.items_per_thread >= 1 && a.blocks >= 1 || throw(ArgumentError(
        "RadixSelect: `items_per_thread` and `blocks` must be positive"))
    # (the global passes stage a tile of keys in local memory, next to their histograms)
    a.items_per_thread <= LOCAL_MEMORY_BUDGET ÷ 2 ÷ (a.block_size * sizeof(select_key_type(T))) ||
        throw(ArgumentError("RadixSelect: a tile of `block_size * items_per_thread` keys of " *
                            "$T must fit in $(LOCAL_MEMORY_BUDGET ÷ 2) bytes"))
    nothing
end
check_select(a::SortSelect, backend, t, T, nr, len) = nothing


# Scratch

function select_scratch(a::RadixSelect, src, len, nslices, nr, dims, rev, backend)
    K = select_key_type(eltype(src))
    t = tuning(select_tuning, backend, eltype(src))
    global_passes = len > t.block_max_len && nslices > 0
    nslices = global_passes ? nslices : 0
    bps = global_passes ? select_bps(a, len, nslices) : 0
    cap = global_passes ? cld(len, COMPACT_FRACTION) : 0
    return (; hist=scratch(UInt32, 2 * 2 * 256 * nslices), kst=scratch(K, KST * nslices),
            ust=scratch(UInt32, UST * nslices), pmin=scratch(K, bps * nslices),
            pmax=scratch(K, bps * nslices), comp=scratch(K, cap * nslices))
end

function select_scratch(a::SortSelect, src, len, nslices, nr, dims, rev, backend)
    sorted = sort_setup(src; backend, alg=a.sort_alg, dims, rev).plan
    return (; temp=scratch(eltype(src), size(src)), ranks=scratch(Int, nr), sort=sorted)
end

# Blocks per slice of the global passes
select_bps(a::RadixSelect, len, nslices) =
    clamp(cld(a.blocks, nslices), 1, cld(len, a.block_size * a.items_per_thread))


# Runs

# (the kernels take the ranks as a tuple of one or two)
select_run!(a::RadixSelect, dst, src, ranks::Vector{Int}, backend, dims, rev, bufs) =
    length(ranks) == 1 ?
        radix_select!(a, dst, src, (ranks[1],), backend, dims, rev, bufs) :
        radix_select!(a, dst, src, (ranks[1], ranks[2]), backend, dims, rev, bufs)

function radix_select!(a::RadixSelect, dst, src, rk::NTuple{NS, Int}, backend, dims, rev,
                       bufs) where {NS}
    T = eltype(src)
    t = tuning(select_tuning, backend, T)
    src_layout = slice_layout(src, dims)
    dst_layout = dims isa Colon ? FlatLayout(NS) : slice_layout(dst, dims)
    len = src_layout.len
    nslices = slice_count(src_layout)
    if len <= t.count_max_len
        B = 256
        span = nextpow(2, len)
        per = B ÷ span
        select_count_kernel!(backend, B)(dst, src, rev, rk, src_layout, dst_layout, Val(span);
                                         ndrange=B * cld(nslices, per))
    elseif len <= t.block_max_len
        B = len <= 2048 ? 256 : 512
        items = nextpow(2, cld(len, B))
        W = scan_subgroup_width(backend, UInt32, B)
        select_block_kernel!(backend, B)(dst, src, rev, rk, src_layout, dst_layout, Val(items),
                                         W; ndrange=B * nslices)
    else
        select_global!(a, dst, src, rk, backend, src_layout, dst_layout, rev, bufs)
    end
    return dst
end

# (always with two ranks, duplicating a single one: the kernels for one rank take more registers,
# and run fewer blocks at a time)
function select_global!(a, dst, src, ranks::NTuple{NR}, backend, src_layout, dst_layout, rev,
                        bufs) where {NR}
    rk = (first(ranks), last(ranks))
    NS = 2
    T = eltype(src)
    K = select_key_type(T)
    len = src_layout.len
    nslices = slice_count(src_layout)
    B = a.block_size
    bps = select_bps(a, len, nslices)
    cap = cld(len, COMPACT_FRACTION)
    W = reduce_subgroup_width(backend, K, B)
    WR = reduce_subgroup_width(backend, K, 256)
    WS = scan_subgroup_width(backend, UInt32, 256)
    (; hist, kst, ust, pmin, pmax, comp) = bufs
    select_init_kernel!(backend, 256)(hist, kst, ust, rk, len; ndrange=256 * nslices)
    hist! = select_hist_kernel!(backend, B)
    resolve! = select_resolve_kernel!(backend, 256)
    for pass in 0:sizeof(K) - 1
        hist!(hist, comp, pmin, pmax, kst, ust, src, rev, src_layout, pass, bps, cap, Val(NS),
              Val(a.items_per_thread), W; ndrange=B * bps * nslices)
        resolve!(dst, hist, pmin, pmax, kst, ust, dst_layout, Val(T), rev, pass, bps, cap, NR,
                 Val(NS), WS, WR; ndrange=256 * nslices)
    end
    return dst
end

function select_run!(a::SortSelect, dst, src, ranks, backend, dims, rev, bufs)
    temp = bufs.temp
    copyto_on!(temp, src, backend)
    s = sort_setup(temp; backend, alg=a.sort_alg, dims, rev)
    sort_run!(s.plan.alg, temp, backend, dims, s.ord, identity, bufs.sort)
    rk = bufs.ranks
    copyto!(rk, ranks)
    max_tasks = one_task(Threads.nthreads(), dst)
    if dims isa Colon
        foreachindex_on(Base.OneTo(length(dst)), backend; max_tasks) do i
            @inbounds dst[i] = temp[rk[i]]
        end
    else
        ci = CartesianIndices(dst)
        d = Int(dims)
        foreachindex_on(Base.OneTo(length(dst)), backend; max_tasks) do i
            @inbounds begin
                I = Tuple(ci[i])
                dst[i] = temp[CartesianIndex(Base.setindex(I, rk[I[d]], d))]
            end
        end
    end
    return dst
end
