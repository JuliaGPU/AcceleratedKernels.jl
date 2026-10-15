function accumulate_nd!(
    op::OP, v::AbstractArray, backend::Backend, alg::Union{SliceScan, CPUThreads.Partitioned};
    src=v,
    entry=identity,
    init,
    neutral,
    dims::Int,
    inclusive::Bool,
    totals::Union{Nothing, AbstractVector}=nothing,
) where {OP}
    # Degenerate cases begin; order of priority matters

    # Nothing to accumulate
    vsizes = size(v)
    if length(v) == 0 || dims > length(vsizes)
        return v
    end
    for s in vsizes
        s == 0 && return v
    end

    # Degenerate cases end

    seed = _scan_first_seed(v, init, neutral)
    if alg isa CPUThreads.Partitioned
        @argcheck src === v
        _accumulate_nd_cpu_sections!(op, v; seed, neutral, dims, inclusive,
                                     max_tasks=alg.max_tasks, min_elems=alg.min_elems)
    else
        n_inner, len, n_outer = _slice_extents(vsizes, dims)
        t = scan_tuning(backend, eltype(v))
        _slice_scan!(op, v, backend, alg, t; src, entry, seed, neutral, inclusive, n_inner, len,
                     n_outer, totals)
    end

    return v
end


# The kernels index `v` linearly, so they step through it with the strides of a column-major array
# of its size (`Base.size_to_strides`), not with `strides(v)`: a wrapper's storage layout, such as a
# `PermutedDimsArray`'s, differs from its linear indices.

function _accumulate_nd_cpu_sections!(
    op, v::AbstractArray;
    seed, neutral, dims, inclusive,
    max_tasks, min_elems,
)
    _, op = _lanefuncs(_Partials(), op, neutral)
    vsizes = size(v)
    vstrides = Base.size_to_strides(1, vsizes...)

    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    # Each thread handles a section of the output array - i.e. reducing along the dims, for
    # multiple output strides
    _foreachindex(1:length_outer, HOST_BACKEND; max_tasks, min_elems) do idst

        @inbounds begin
            # Compute the base index in v for this outer axis
            input_base_idx = 0
            tmp = idst
            KernelAbstractions.Extras.@unroll for i in 1:ndims
                if i != dims
                    input_base_idx += (tmp % vsizes[i]) * vstrides[i]
                    tmp = tmp ÷ vsizes[i]
                end
            end

            # Go over each element in the accumulated dimension
            running = seed
            for i in 0:length_dims - 1
                v_idx = input_base_idx + i * vstrides[dims]
                x = _lift(neutral, v[v_idx + 1])
                if inclusive
                    running = op(running, x)
                    v[v_idx + 1] = _unlane(running)
                else
                    v[v_idx + 1] = _unlane(running)
                    running = op(running, x)
                end
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
# reduced to its total, the totals of each slice are scanned (exclusively, from the scan's seed)
# into the carry of each chunk, and each chunk is scanned again from its carry. The totals form
# an `(n_inner, nchunks, n_outer)` array, scanned along its second dimension by the same means.
# Reducing first, rather than scanning each chunk and adding the carries afterwards, reads the
# source twice but writes it once, and leaves the source intact until the last pass, so a scan
# in place needs no other scratch. Every pass reads the elements as `entry(src[i])`, so a scan
# out of place needs no copy either. Chunks and carries keep the elements' order, so `op` need
# not be commutative.

# The extents `(n_inner, len, n_outer)` of an array of size `sz` scanned along `dims`
_slice_extents(sz::Dims{N}, dims::Int) where {N} =
    (Base.prod(ntuple(d -> d < dims ? sz[d] : 1, Val(N))), sz[dims],
     Base.prod(ntuple(d -> d > dims ? sz[d] : 1, Val(N))))

# The launch shape of a scan of `n_inner * n_outer` slices of `len` elements, whose partial
# results have type `S`: the kind of kernel (`:rows`, `:columns` or `:tiles`), its block size, the
# number of chunks per slice, and the elements (for `:columns`) or tiles (for `:tiles`) per chunk
# or the size of the local memory (for `:rows`)
#
# A split makes chunks of at least two elements (but the last), so that the totals scanned next
# are fewer than the elements, whatever the settings.
function _slice_shape(n_inner, len, n_outer, a::SliceScan, t::ScanTuning, ::Type{S}) where {S}
    nslices = n_inner * n_outer
    max_block, items = a.block_size::Int, a.items_per_thread::Int
    if n_inner == 1 && nslices >= t.target_blocks
        # The most slices whose elements fit the local-memory budget, in a block of at least 32
        # threads (with fewer slices, a block per slice keeps more threads busy)
        rows = _slice_rows_max(t, S) ÷ len
        if rows >= min(32, max_block)
            return (:rows, min(prevpow(2, rows), max_block), 1, _slice_rows_max(t, S))
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
_slice_rows_max(t::ScanTuning, ::Type{S}) where {S} = t.local_mem_bytes ÷ max(sizeof(S), 1)

# The number of chunk totals a scan of slices of `len` elements stores, at every level
function _slice_scratch(n_inner, len, n_outer, a::SliceScan, t::ScanTuning, ::Type{S}) where {S}
    _, _, nchunks, _ = _slice_shape(n_inner, len, n_outer, a, t, S)
    nchunks == 1 && return 0
    n = n_inner * nchunks * n_outer
    return n + _slice_scratch(n_inner, nchunks, n_outer, a, t, S)
end
_slice_scratch(sz::Dims, dims::Int, a::SliceScan, t::ScanTuning, ::Type{S}) where {S} =
    dims > length(sz) || Base.prod(sz; init=1) == 0 ? 0 :
    _slice_scratch(_slice_extents(sz, dims)..., a, t, S)

# Scan the slices of the elements `entry(src[i])` into `v` (`src` may be `v`), as an
# `(n_inner, len, n_outer)` array scanned along its second dimension; `totals` holds at least
# `_slice_scratch` chunk totals after `offset`. (The levels of totals are views of `totals`
# itself, not of each other, so that every level has the same types.)
function _slice_scan!(op::OP, v, backend, a::SliceScan, t::ScanTuning;
                      src=v, entry=identity, seed, neutral, inclusive, n_inner, len, n_outer,
                      totals, offset=0) where {OP}
    kind, block_size, nchunks, per_chunk =
        _slice_shape(n_inner, len, n_outer, a, t, typeof(neutral))
    items = a.items_per_thread
    nslices = n_inner * n_outer

    if nchunks == 1
        _slice_pass!(kind, backend, block_size, nslices, v, src, entry, op, seed, neutral,
                     nothing, nothing, inclusive, n_inner, len, 1, per_chunk, items)
        return v
    end

    n = nslices * nchunks
    @argcheck length(totals) >= offset + n
    chunk_totals = @view totals[offset + 1:offset + n]
    # Each chunk's total, without the seed
    _slice_pass!(kind, backend, block_size, nslices, v, src, entry, op, neutral, neutral, nothing,
                 chunk_totals, inclusive, n_inner, len, nchunks, per_chunk, items)
    # Each chunk's carry: the seed, then the totals of the chunks before it
    _slice_scan!(op, chunk_totals, backend, a, t; seed, neutral, inclusive=false, n_inner,
                 len=nchunks, n_outer, totals, offset=offset + n)
    # Each chunk, from its carry
    _slice_pass!(kind, backend, block_size, nslices, v, src, entry, op, seed, neutral,
                 chunk_totals, nothing, inclusive, n_inner, len, nchunks, per_chunk, items)
    return v
end

# One pass over all chunks: scan each chunk of `v` from `seed`, or from its carry in `carries`, or
# when `totals` is given store each chunk's total instead. (The compile-time sizes become `Val`s
# only at their launch, which takes a runtime dispatch, so that the rest of the scan infers.)
function _slice_pass!(kind, backend, block_size, nslices, v, src, entry, op::OP, seed, neutral,
                      carries, totals, inclusive, n_inner, len, nchunks, per_chunk,
                      items) where {OP}
    if kind === :rows
        kernel! = _slice_scan_rows!(backend, block_size)
        kernel!(v, src, entry, op, seed, neutral, inclusive, len, nslices, Val(per_chunk);
                ndrange=(block_size * cld(nslices, block_size),))
    elseif kind === :columns
        nthreads = nslices * nchunks
        kernel! = _slice_scan_columns!(backend, block_size)
        kernel!(v, src, entry, op, seed, neutral, carries, totals, inclusive, n_inner, len,
                nchunks, per_chunk, nthreads; ndrange=(block_size * cld(nthreads, block_size),))
    else
        kernel! = _slice_scan_tiles!(backend, block_size)
        kernel!(v, src, entry, op, seed, neutral, carries, totals, inclusive, n_inner, len,
                nchunks, per_chunk, Val(items); ndrange=(block_size * nslices * nchunks,))
    end
    return nothing
end

# One thread per contiguous slice: block `b` stages slices `b * block_size` onwards, the
# `block_size * len` consecutive elements from `b * block_size * len`, in local memory of `MAX`
# elements (`_slice_rows_max`, the same for every `len`)
@kernel inbounds=true cpu=false unsafe_indices=true function _slice_scan_rows!(
    v, src, entry, op, seed, neutral, inclusive, len, nslices, ::Val{MAX},
) where {MAX}
    @uniform block_size = @groupsize()[1]
    rows = @localmem typeof(neutral) (MAX,)
    _, op = _lanefuncs(_Partials(), op, neutral)

    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1
    offset = iblock * block_size * len
    n = min(block_size * len, nslices * len - offset)

    p = ithread
    while p < n
        rows[p + 1] = _lift(neutral, entry(src[offset + p + 1]))
        p += block_size
    end
    @synchronize()

    if ithread * len < n
        running = seed
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
        _store!(v, offset + p + 1, rows[p + 1])
        p += block_size
    end
end

# One thread per chunk of a slice; thread `t` takes chunk `c` of slice `(i, o)` with
# `t = i + n_inner * (c + nchunks * o)`, the chunk's position in the array of totals
@kernel inbounds=true cpu=false unsafe_indices=true function _slice_scan_columns!(
    v, src, entry, op, seed, neutral, carries, totals, inclusive,
    n_inner, len, nchunks, chunk_len, nthreads,
)
    _, op = _lanefuncs(_Partials(), op, neutral)
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

        running = isnothing(carries) ? seed : carries[tid + 1]
        if isnothing(totals)
            for k in k0:k1 - 1
                idx = base + k * n_inner + 1
                x = _lift(neutral, entry(src[idx]))
                if inclusive
                    running = op(running, x)
                    _store!(v, idx, running)
                else
                    _store!(v, idx, running)
                    running = op(running, x)
                end
            end
        else
            for k in k0:k1 - 1
                running = op(running, _lift(neutral, entry(src[base + k * n_inner + 1])))
            end
            totals[tid + 1] = running
        end
    end
end

# One block per chunk of a slice, scanning `tiles_per_chunk` tiles of the slice in turn; block `b`
# takes chunk `c` of slice `(i, o)` with `b = c + nchunks * (i + n_inner * o)`, and the chunk's
# total or carry is at `i + n_inner * (c + nchunks * o)` in the array of totals
@kernel inbounds=true cpu=false unsafe_indices=true function _slice_scan_tiles!(
    v, src, entry, op, seed, neutral, carries, totals, inclusive,
    n_inner, len, nchunks, tiles_per_chunk, ::Val{ITEMS},
) where {ITEMS}
    @uniform block_size = @groupsize()[1]
    tile = @localmem typeof(neutral) (block_size * ITEMS,)
    thread_totals = @localmem typeof(neutral) (block_size,)
    _, op = _lanefuncs(_Partials(), op, neutral)

    iblock = @index(Group, Linear) - 1
    ithread = @index(Local, Linear) - 1
    c = iblock % nchunks
    s = iblock ÷ nchunks
    i = s % n_inner
    o = s ÷ n_inner
    base = i + n_inner * len * o
    tile_len = block_size * ITEMS
    k_end = min((c + 1) * tiles_per_chunk * tile_len, len)

    running = isnothing(carries) ? seed : carries[i + n_inner * (c + nchunks * o) + 1]
    k_tile = c * tiles_per_chunk * tile_len
    while k_tile < k_end
        # Load the tile in striped order
        j = 0
        while j < ITEMS
            p = j * block_size + ithread
            k = k_tile + p
            tile[p + 1] = k < k_end ? _lift(neutral, entry(src[base + k * n_inner + 1])) :
                                      neutral
            j += 1
        end
        @synchronize()

        # Scan each thread's run of consecutive elements, then the runs' totals
        run = ithread * ITEMS
        acc = neutral
        j = 0
        while j < ITEMS
            acc = op(acc, tile[run + j + 1])
            tile[run + j + 1] = acc
            j += 1
        end
        thread_totals[ithread + 1] = acc
        # (the carry is combined with each prefix last, as in a sequential scan from the carry
        # it would be first: a floating-point prefix then rounds at its own magnitude, not at
        # the carry's)
        thread_prefix, block_total = block_exclusive_scan!(
            @context, op, thread_totals, neutral, block_size, ithread,
        )

        if isnothing(totals)
            prev = neutral
            j = 0
            while j < ITEMS
                r = tile[run + j + 1]
                tile[run + j + 1] = op(running, op(thread_prefix, inclusive ? r : prev))
                prev = r
                j += 1
            end
            @synchronize()

            # Store the tile in striped order
            j = 0
            while j < ITEMS
                p = j * block_size + ithread
                k = k_tile + p
                if k < k_end
                    _store!(v, base + k * n_inner + 1, tile[p + 1])
                end
                j += 1
            end
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
