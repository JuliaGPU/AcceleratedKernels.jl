function accumulate_nd!(
    op::OP, v::AbstractArray, backend::Backend, alg::SliceScan;
    src=v,
    load=identity,
    init,
    seed,
    dims::Int,
    inclusive::Bool,
    totals::Union{Nothing, AbstractVector}=nothing,
) where {OP}
    n_inner, len, n_outer = slice_extents(size(v), dims)
    t = scan_tuning(backend, eltype(v))
    slice_scan!(op, v, backend, alg, t; src, load, start=scan_start(v, init, seed), seed,
                inclusive, n_inner, len, n_outer, totals)
    return v
end


# Scan each slice along `dims` in one task, as `slice_extents` lays them out
function accumulate_nd_threads!(
    op, v::AbstractArray;
    start, seed, dims, inclusive,
    max_tasks, min_elems,
)
    _, op = seeded_ops(PartialInputs(), op, seed)
    n_inner, len, n_outer = slice_extents(size(v), dims)
    foreachindex_on(1:n_inner * n_outer, HOST_BACKEND; max_tasks, min_elems) do islice
        base = (islice - 1) % n_inner + n_inner * len * ((islice - 1) ÷ n_inner) + 1
        running = start
        @inbounds for k in 0:len - 1
            idx = base + k * n_inner
            x = lift(seed, v[idx])
            if inclusive
                running = op(running, x)
                store_partial!(v, idx, running)
            else
                store_partial!(v, idx, running)
                running = op(running, x)
            end
        end
    end
    v
end


# GPU scans along `dims`
#
# An array scanned along `dims` is a three-dimensional array of size `(n_inner, len, n_outer)`:
# `len` elements per slice, consecutive slices `n_inner` apart in linear order and in blocks of
# `n_inner` that are `n_inner * len` apart. Element `k` of slice `(i, o)` (zero-based) is at the
# linear index `i + n_inner * (k + len * o)`.
#
# The memory layout decides which threads walk a slice:
#   - `:rows`: short contiguous slices (`n_inner == 1`), one thread per slice: a block loads its
#     slices, which are consecutive in memory, into local memory with coalesced loads, each thread
#     scans its own slice there, and the block stores them back alike.
#   - `:columns`: one thread per slice, stepping along it in memory. Neighbouring threads scan
#     neighbouring slices, so their loads are coalesced when `n_inner` is large enough.
#   - `:tiles`: a block per slice, loading tiles of `block_size * items_per_thread` consecutive
#     elements of it. Its loads are coalesced when the slice is contiguous, and for a small
#     `n_inner` the other slices' blocks read the rest of the same cache lines.
#
# When that leaves too few threads or blocks to fill the device, each slice of a large enough
# array is split into chunks that are scanned independently, in three passes: each chunk is
# reduced to its total, the totals of each slice are scanned (exclusively, from the scan's start)
# into the carry of each chunk, and each chunk is scanned again from its carry. The totals form
# an `(n_inner, nchunks, n_outer)` array, scanned along its second dimension by the same means.
# Reducing first, rather than scanning each chunk and adding the carries afterwards, reads the
# source twice but writes it once, and leaves the source intact until the last pass, so a scan
# in place needs no other scratch.

# The extents `(n_inner, len, n_outer)` of an array of size `sz` scanned along `dims`
# (the scans index arrays linearly, so these are the extents of a column-major array of that size,
# whatever the storage layout of a wrapper such as a `PermutedDimsArray`)
slice_extents(sz::Dims{N}, dims::Int) where {N} =
    (Base.prod(ntuple(d -> d < dims ? sz[d] : 1, Val(N))), sz[dims],
     Base.prod(ntuple(d -> d > dims ? sz[d] : 1, Val(N))))

# The launch shape of a scan of `n_inner * n_outer` slices of `len` elements, whose partial
# results have type `S`: the kind of kernel (`:rows`, `:columns` or `:tiles`), its block size, the
# number of chunks per slice, and the elements (for `:columns`) or tiles (for `:tiles`) per chunk
# or the size of the local memory (for `:rows`)
#
# A split makes chunks of at least two elements (but the last), so that the totals scanned next
# are fewer than the elements, whatever the settings.
function slice_shape(n_inner, len, n_outer, a::SliceScan, t::ScanTuning, ::Type{S}) where {S}
    nslices = n_inner * n_outer
    max_block, items = a.block_size::Int, a.items_per_thread::Int
    if n_inner == 1 && nslices >= t.target_blocks
        # The most slices whose elements fit the local-memory budget, in a block of at least 32
        # threads (with fewer slices, a block per slice keeps more threads busy)
        rows = slice_rows_max(t, S) ÷ len
        if rows >= min(32, max_block)
            return (:rows, min(prevpow(2, rows), max_block), 1, slice_rows_max(t, S))
        end
    end
    # Neighbouring slices get a thread each: whole when there are enough of them or they are
    # short, else split into chunks, but only in large arrays: in small ones, the passes that
    # splitting adds would take longer, and a block per slice shares the work instead
    target = t.target_blocks * max_block
    if len <= t.slice_serial_len ||
       n_inner >= t.slice_column_inner &&
       (nslices >= target || len <= t.slice_columns_len || nslices * len >= t.slice_split_min)
        nchunks = nslices >= target || len <= t.slice_columns_len ? 1 :
                  clamp(cld(target, nslices), 1, cld(len, t.slice_min_chunk))
        chunk_len = max(cld(len, nchunks), 2)
        return (:columns, max_block, cld(len, chunk_len), chunk_len)
    end
    # Other slices get a block each, the smallest whose tile holds a whole slice (up to the
    # algorithm's block size), split into chunks when there are too few slices to fill the
    # device and a block would scan more than `slice_split_tiles` tiles on its own
    block_size = clamp(nextpow(2, cld(len, items)), min(32, max_block), max_block)
    ntiles = cld(len, block_size * items)
    nchunks = nslices >= t.target_blocks || ntiles <= t.slice_split_tiles ? 1 :
              clamp(cld(t.target_blocks, nslices), 1, ntiles)
    tiles_per_chunk = max(cld(ntiles, nchunks), cld(2, block_size * items))
    return (:tiles, block_size, cld(ntiles, tiles_per_chunk), tiles_per_chunk)
end

# The elements of type `S` that the local memory of a `:rows` block holds
slice_rows_max(t::ScanTuning, ::Type{S}) where {S} = t.local_mem_bytes ÷ max(sizeof(S), 1)

# The number of chunk totals a scan of slices of `len` elements stores, at every level
function slice_scratch(n_inner, len, n_outer, a::SliceScan, t::ScanTuning, ::Type{S}) where {S}
    _, _, nchunks, _ = slice_shape(n_inner, len, n_outer, a, t, S)
    nchunks == 1 && return 0
    n = n_inner * nchunks * n_outer
    return n + slice_scratch(n_inner, nchunks, n_outer, a, t, S)
end
slice_scratch(sz::Dims, dims::Int, a::SliceScan, t::ScanTuning, ::Type{S}) where {S} =
    dims > length(sz) || Base.prod(sz; init=1) == 0 ? 0 :
    slice_scratch(slice_extents(sz, dims)..., a, t, S)

# Scan the slices of the elements `load(src[i])` into `v` (`src` may be `v`), as an
# `(n_inner, len, n_outer)` array scanned along its second dimension; `totals` holds at least
# `slice_scratch` chunk totals after `offset`. (The levels of totals are views of `totals`
# itself, not of each other, so that every level has the same types.)
function slice_scan!(op::OP, v, backend, a::SliceScan, t::ScanTuning;
                     src=v, load=identity, start, seed, inclusive, n_inner, len, n_outer,
                     totals, offset=0) where {OP}
    kind, block_size, nchunks, per_chunk =
        slice_shape(n_inner, len, n_outer, a, t, typeof(seed))
    items = a.items_per_thread
    nslices = n_inner * n_outer

    if nchunks == 1
        slice_pass!(kind, backend, block_size, nslices, v, src, load, op, start, seed,
                    nothing, nothing, inclusive, n_inner, len, 1, per_chunk, items)
        return v
    end

    n = nslices * nchunks
    chunk_totals = @view totals[offset + 1:offset + n]
    # Each chunk's total, without the start
    slice_pass!(kind, backend, block_size, nslices, v, src, load, op, seed, seed, nothing,
                chunk_totals, inclusive, n_inner, len, nchunks, per_chunk, items)
    # Each chunk's carry: the start, then the totals of the chunks before it
    slice_scan!(op, chunk_totals, backend, a, t; start, seed, inclusive=false, n_inner,
                len=nchunks, n_outer, totals, offset=offset + n)
    # Each chunk, from its carry
    slice_pass!(kind, backend, block_size, nslices, v, src, load, op, start, seed,
                chunk_totals, nothing, inclusive, n_inner, len, nchunks, per_chunk, items)
    return v
end

# One pass over all chunks: scan each chunk of `v` from `start`, or from its carry in `carries`, or
# when `totals` is given store each chunk's total instead. (The compile-time sizes become `Val`s
# only at their launch, which takes a runtime dispatch, so that the rest of the scan infers.)
function slice_pass!(kind, backend, block_size, nslices, v, src, load, op::OP, start, seed,
                     carries, totals, inclusive, n_inner, len, nchunks, per_chunk,
                     items) where {OP}
    if kind === :rows
        kernel! = slice_rows_kernel!(backend, block_size)
        kernel!(v, src, load, op, start, seed, inclusive, len, nslices, Val(per_chunk);
                ndrange=(block_size * cld(nslices, block_size),))
    elseif kind === :columns
        nthreads = nslices * nchunks
        kernel! = slice_columns_kernel!(backend, block_size)
        kernel!(v, src, load, op, start, seed, carries, totals, inclusive, n_inner, len,
                nchunks, per_chunk, nthreads; ndrange=(block_size * cld(nthreads, block_size),))
    else
        kernel! = slice_tiles_kernel!(backend, block_size)
        kernel!(v, src, load, op, start, seed, carries, totals, inclusive, n_inner, len,
                nchunks, per_chunk, Val(items),
                scan_subgroup_width(backend, typeof(seed), block_size);
                ndrange=(block_size * nslices * nchunks,))
    end
    return nothing
end

# One thread per contiguous slice: block `b` stages slices `b * block_size` onwards, the
# `block_size * len` consecutive elements from `b * block_size * len`, in local memory of `MAX`
# elements (`slice_rows_max`, the same for every `len`)
@kernel inbounds=true cpu=false unsafe_indices=true function slice_rows_kernel!(
    v, src, load, op, start, seed, inclusive, len, nslices, ::Val{MAX},
) where {MAX}
    @uniform block_size = @groupsize()[1]
    rows = @localmem typeof(seed) (MAX,)
    _, op = seeded_ops(PartialInputs(), op, seed)

    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1
    offset = iblock * block_size * len
    n = min(block_size * len, nslices * len - offset)

    p = ithread
    while p < n
        rows[p + 1] = lift(seed, load(src[offset + p + 1]))
        p += block_size
    end
    @synchronize()

    if ithread * len < n
        running = start
        for k in ithread * len:(ithread + 1) * len - 1
            x = rows[k + 1]
            if inclusive
                running = op(running, x)
                rows[k + 1] = running
            else
                rows[k + 1] = running
                running = op(running, x)
            end
        end
    end
    @synchronize()

    p = ithread
    while p < n
        store_partial!(v, offset + p + 1, rows[p + 1])
        p += block_size
    end
end

# One thread per chunk of a slice; thread `t` takes chunk `c` of slice `(i, o)` with
# `t = i + n_inner * (c + nchunks * o)`, the chunk's position in the array of totals
@kernel inbounds=true cpu=false unsafe_indices=true function slice_columns_kernel!(
    v, src, load, op, start, seed, carries, totals, inclusive,
    n_inner, len, nchunks, chunk_len, nthreads,
)
    _, op = seeded_ops(PartialInputs(), op, seed)
    block_size = @groupsize()[1]
    tid = (@index(Group, Linear) - 1) * block_size + (@index(Local, Linear) - 1)
    if tid < nthreads
        i = tid % n_inner
        r = tid ÷ n_inner
        c = r % nchunks
        o = r ÷ nchunks
        base = i + n_inner * len * o
        k0 = c * chunk_len
        k1 = min(k0 + chunk_len, len)

        running = isnothing(carries) ? start : carries[tid + 1]
        if isnothing(totals)
            for k in k0:k1 - 1
                idx = base + k * n_inner + 1
                x = lift(seed, load(src[idx]))
                if inclusive
                    running = op(running, x)
                    store_partial!(v, idx, running)
                else
                    store_partial!(v, idx, running)
                    running = op(running, x)
                end
            end
        else
            for k in k0:k1 - 1
                running = op(running, lift(seed, load(src[base + k * n_inner + 1])))
            end
            totals[tid + 1] = running
        end
    end
end

# One block per chunk of a slice, scanning `tiles_per_chunk` tiles of the slice in turn; block `b`
# takes chunk `c` of slice `(i, o)` with `b = c + nchunks * (i + n_inner * o)`, and the chunk's
# total or carry is at `i + n_inner * (c + nchunks * o)` in the array of totals
@kernel inbounds=true cpu=false unsafe_indices=true function slice_tiles_kernel!(
    v, src, load::L, op::OP, start, seed, carries, totals, inclusive,
    n_inner, len, nchunks, tiles_per_chunk, ::Val{ITEMS}, subgroups::Val,
) where {L, OP, ITEMS}
    @uniform block_size = @groupsize()[1]
    tile = @localmem typeof(seed) (block_size * ITEMS,)
    thread_totals = @localmem typeof(seed) (block_size,)
    _, op = seeded_ops(PartialInputs(), op, seed)

    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1
    c = iblock % nchunks
    s = iblock ÷ nchunks
    i = s % n_inner
    o = s ÷ n_inner
    base = i + n_inner * len * o
    tile_len = block_size * ITEMS
    k_end = min((c + 1) * tiles_per_chunk * tile_len, len)

    running = isnothing(carries) ? start : carries[i + n_inner * (c + nchunks * o) + 1]
    k_tile = c * tiles_per_chunk * tile_len
    while k_tile < k_end
        index = k -> base + k * n_inner + 1
        tile_load!(tile, src, load, seed, index, k_tile, k_end, ithread, block_size, Val(ITEMS))
        @synchronize()

        # Scan each thread's run, then the runs' totals
        # (the carry is combined with each prefix last, as in a sequential scan from the carry
        # it would be first: a floating-point prefix then rounds at its own magnitude, not at
        # the carry's)
        thread_scan!(op, tile, thread_totals, seed, ithread, Val(ITEMS))
        thread_prefix, block_total = block_exclusive_scan!(
            @context, op, thread_totals, seed, block_size, ithread, subgroups,
        )
        if isnothing(totals)
            tile_prefix!(op, tile, thread_prefix, running, seed, inclusive, ithread, Val(ITEMS))
            @synchronize()
            tile_store!(v, tile, index, k_tile, k_end, ithread, block_size, Val(ITEMS))
        end
        running = op(running, block_total)

        # Every thread is done with the tile before the next one is loaded
        @synchronize()
        k_tile += tile_len
    end

    if !isnothing(totals) && ithread == 0
        totals[i + n_inner * (c + nchunks * o) + 1] = running
    end
end
