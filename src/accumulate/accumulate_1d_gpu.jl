# Register-raking block scan with striped loads and stores, of the elements `entry(src[i])` into
# `v` (`src` may be `v`). `neutral` seeds partial results (an empty `_Lane` when `op` has no known
# neutral element), and `seed` is the first block's. With `block_seeds`, the inclusive scan of
# the blocks' totals, block `b` is seeded with `block_seeds[b]`, the reduction of the blocks
# before it; else with `neutral`, to receive its carry from another kernel.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_block!(
    op, v, src, entry, seed, neutral,
    inclusive,
    prefixes, block_seeds,
    ::Val{ITEMS}, subgroups::Val,
) where ITEMS
    # `block_size` is a power of two.
    @uniform block_size = @groupsize()[1]
    tile = @localmem typeof(neutral) (block_size * ITEMS,)
    thread_totals = @localmem typeof(neutral) (block_size,)
    _, op = _lanefuncs(_Partials(), op, neutral)

    # Internal indices are zero-based; add one only when indexing arrays.
    len = length(v)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    # Load a tile in striped order.
    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        tile[p + 0x1] = gi < len ? _lift(neutral, entry(src[gi + 0x1])) : neutral
        j += 1
    end
    @synchronize()

    # Scan each thread's blocked run.
    run = ithread * ITEMS
    acc = neutral
    k = 0
    while k < ITEMS
        acc = op(acc, tile[run + k + 0x1])
        tile[run + k + 0x1] = acc
        k += 1
    end
    thread_totals[ithread + 0x1] = acc

    # Scan the per-thread totals, from the block's seed
    block_seed = iblock == 0x0 ? seed : isnothing(block_seeds) ? neutral : block_seeds[iblock]
    thread_prefix, block_total = block_exclusive_scan!(
        @context, op, thread_totals, block_seed, block_size, ithread, subgroups,
    )

    prev = neutral
    k = 0
    while k < ITEMS
        r = tile[run + k + 0x1]
        tile[run + k + 0x1] = inclusive ? op(thread_prefix, r) : op(thread_prefix, prev)
        prev = r
        k += 1
    end
    @synchronize()

    if ithread == 0x0 && !isnothing(prefixes)
        prefixes[iblock + 0x1] = block_total
    end

    # Store the tile in striped order.
    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        if gi < len
            _store!(v, gi + 0x1, tile[p + 0x1])
        end
        j += 1
    end
end


# The total of each tile of `block_size * ITEMS` elements `entry(src[i])` into `totals`, the
# first tile's from `seed`. Each thread reduces a run of consecutive elements, in order, and the
# block then combines the runs; no tile is staged in local memory, as nothing is stored.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_tile_totals!(
    op, totals, src, entry, seed, neutral, ::Val{ITEMS}, subgroups::Val,
) where ITEMS
    @uniform block_size = @groupsize()[1]
    thread_totals = @localmem typeof(neutral) (block_size,)
    _, op = _lanefuncs(_Partials(), op, neutral)

    len = length(src)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    run = iblock * block_size * ITEMS + ithread * ITEMS

    acc = neutral
    k = 0
    while k < ITEMS
        gi = run + k
        if gi < len
            acc = op(acc, _lift(neutral, entry(src[gi + 0x1])))
        end
        k += 1
    end
    thread_totals[ithread + 0x1] = acc

    _, block_total = block_exclusive_scan!(
        @context, op, thread_totals, iblock == 0x0 ? seed : neutral, block_size, ithread,
        subgroups,
    )
    if ithread == 0x0
        totals[iblock + 0x1] = block_total
    end
end

# Add to each tile after the first the reduction of the tiles before it, from the inclusive scan
# of the tiles' totals.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_add_prefixes!(
    op, v, totals, neutral, inclusive, ::Val{ITEMS},
) where ITEMS
    _, op = _lanefuncs(_Partials(), op, neutral)
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS
    prefix = totals[iblock]

    # An exclusive tile starts with its seed, `neutral`, which may be an empty lane that cannot
    # be stored: its first element is the prefix itself
    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = if !inclusive && gi == block_offset
                _lower(eltype(v), prefix)
            else
                _lower(eltype(v), op(prefix, _lift(neutral, v[gi + 0x1])))
            end
        end
        j += 1
    end
end

# The first block's seed: `init` in the element type, else the neutral seed
_scan_first_seed(v, init, neutral) = _lift(neutral, convert(eltype(v), init))
_scan_first_seed(v, ::_NoInit, neutral) = neutral


# The length of the scratch of `ScanPrefixes` for `n` elements: the totals of the tiles, of the
# tiles of those totals, and so on until a single tile.
function _scan_prefixes_length(n, alg::ScanPrefixes)
    elems_per_block = alg.block_size * alg.items_per_thread
    len = 0
    num_blocks = cld(n, elems_per_block)
    while num_blocks > 1
        len += num_blocks
        num_blocks = cld(num_blocks, elems_per_block)
    end
    return len
end

# ScanPrefixes algorithm
function accumulate_1d_gpu!(
    op, v::AbstractArray, backend, alg::ScanPrefixes;
    src=v,
    entry=identity,
    init,
    neutral,
    inclusive::Bool,
    prefixes::Union{Nothing, AbstractArray},
)
    block_size, items_per_thread = alg.block_size, alg.items_per_thread

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    elems_per_block = block_size * items_per_thread
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block
    items = Val(items_per_thread)
    seed = _scan_first_seed(v, init, neutral)

    subgroups = _scan_subgroups(backend, typeof(neutral), block_size)
    kernel! = _accumulate_block!(backend, block_size)
    if num_blocks == 1
        kernel!(op, v, src, entry, seed, neutral, inclusive, nothing, nothing, items,
                subgroups, ndrange=block_size)
        return v
    end

    # The tiles' totals are scanned the same way, with the rest of `prefixes` for the totals'
    # totals (from `neutral`, as the first total includes `seed`)
    totals = view(prefixes, 1:num_blocks)
    scan_totals!() =
        accumulate_1d_gpu!(op, totals, backend, alg; init=_NoInit(), neutral, inclusive=true,
                           prefixes=view(prefixes, num_blocks+1:length(prefixes)))
    if scan_tuning(backend, eltype(v)).reduce_then_scan
        # Reduce the tiles, then scan them from their prefix: this reads the source twice and
        # writes the result once
        _accumulate_tile_totals!(backend, block_size)(op, totals, src, entry, seed, neutral,
                                                      items, subgroups,
                                                      ndrange=num_blocks * block_size)
        scan_totals!()
        kernel!(op, v, src, entry, seed, neutral, inclusive, nothing, totals, items,
                subgroups, ndrange=num_blocks * block_size)
    else
        # Scan the tiles, then add their prefix: this reads and writes the result twice
        kernel!(op, v, src, entry, seed, neutral, inclusive, totals, nothing, items,
                subgroups, ndrange=num_blocks * block_size)
        scan_totals!()
        _accumulate_add_prefixes!(backend, block_size)(op, v, totals, neutral, inclusive, items,
                                                       ndrange=(num_blocks - 1) * block_size)
    end
    return v
end
