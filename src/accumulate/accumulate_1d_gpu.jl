# Register-raking block scan with striped loads and stores, of the elements `load(src[i])` into
# `v` (`src` may be `v`). `seed` seeds partial results (an unset `Optional` when `op` has no known
# neutral element), and `start` is the first block's starting value. With `tile_prefixes`, the
# inclusive scan of the blocks' totals, block `b` starts from `tile_prefixes[b]`, the reduction of
# the blocks before it; else from `seed`, to receive its carry from another kernel. With
# `aggregates`, each block stores its total there.
@kernel cpu=false inbounds=true unsafe_indices=true function scan_tile_kernel!(
    op, v, src, load, start, seed,
    inclusive,
    aggregates, tile_prefixes,
    ::Val{ITEMS}, subgroups::Val,
) where ITEMS
    # `block_size` is a power of two.
    @uniform block_size = @groupsize()[1]
    tile = @localmem typeof(seed) (block_size * ITEMS,)
    thread_totals = @localmem typeof(seed) (block_size,)
    _, op = seeded_ops(PartialInputs(), op, seed)

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
        tile[p + 0x1] = gi < len ? lift(seed, load(src[gi + 0x1])) : seed
        j += 1
    end
    @synchronize()

    # Scan each thread's blocked run.
    run = ithread * ITEMS
    acc = seed
    k = 0
    while k < ITEMS
        acc = op(acc, tile[run + k + 0x1])
        tile[run + k + 0x1] = acc
        k += 1
    end
    thread_totals[ithread + 0x1] = acc

    # Scan the per-thread totals, from the block's start
    block_start = iblock == 0x0 ? start : isnothing(tile_prefixes) ? seed : tile_prefixes[iblock]
    thread_prefix, block_total = block_exclusive_scan!(
        @context, op, thread_totals, block_start, block_size, ithread, subgroups,
    )

    prev = seed
    k = 0
    while k < ITEMS
        r = tile[run + k + 0x1]
        tile[run + k + 0x1] = inclusive ? op(thread_prefix, r) : op(thread_prefix, prev)
        prev = r
        k += 1
    end
    @synchronize()

    if ithread == 0x0 && !isnothing(aggregates)
        aggregates[iblock + 0x1] = block_total
    end

    # Store the tile in striped order.
    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        if gi < len
            store_partial!(v, gi + 0x1, tile[p + 0x1])
        end
        j += 1
    end
end


# The total of each tile of `block_size * ITEMS` elements `load(src[i])` into `totals`, the
# first tile's from `start`. Each thread reduces a run of consecutive elements, in order, and the
# block then combines the runs; no tile is staged in local memory, as nothing is stored.
@kernel cpu=false inbounds=true unsafe_indices=true function tile_aggregates_kernel!(
    op, totals, src, load, start, seed, ::Val{ITEMS}, subgroups::Val,
) where ITEMS
    @uniform block_size = @groupsize()[1]
    thread_totals = @localmem typeof(seed) (block_size,)
    _, op = seeded_ops(PartialInputs(), op, seed)

    len = length(src)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    run = iblock * block_size * ITEMS + ithread * ITEMS

    acc = seed
    k = 0
    while k < ITEMS
        gi = run + k
        if gi < len
            acc = op(acc, lift(seed, load(src[gi + 0x1])))
        end
        k += 1
    end
    thread_totals[ithread + 0x1] = acc

    _, block_total = block_exclusive_scan!(
        @context, op, thread_totals, iblock == 0x0 ? start : seed, block_size, ithread,
        subgroups,
    )
    if ithread == 0x0
        totals[iblock + 0x1] = block_total
    end
end

# Add to each tile after the first the reduction of the tiles before it, from the inclusive scan
# of the tiles' totals.
@kernel cpu=false inbounds=true unsafe_indices=true function add_tile_prefixes_kernel!(
    op, v, totals, seed, inclusive, ::Val{ITEMS},
) where ITEMS
    _, op = seeded_ops(PartialInputs(), op, seed)
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS
    prefix = totals[iblock]

    # An exclusive tile starts from `seed`, which may be an unset `Optional` that
    # cannot be stored: its first element is the prefix itself
    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = if !inclusive && gi == block_offset
                lower(eltype(v), prefix)
            else
                lower(eltype(v), op(prefix, lift(seed, v[gi + 0x1])))
            end
        end
        j += 1
    end
end

# The length of the scratch of `ScanPrefixes` for `n` elements: the totals of the tiles, of the
# tiles of those totals, and so on until a single tile.
function prefixes_length(n, alg::ScanPrefixes)
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
    load=identity,
    init,
    seed,
    inclusive::Bool,
    prefixes::Union{Nothing, AbstractArray},
)
    block_size, items_per_thread = alg.block_size, alg.items_per_thread
    num_blocks = cld(length(v), block_size * items_per_thread)
    items = Val(items_per_thread)
    start = scan_start(v, init, seed)

    subgroups = scan_subgroup_width(backend, typeof(seed), block_size)
    kernel! = scan_tile_kernel!(backend, block_size)
    if num_blocks == 1
        kernel!(op, v, src, load, start, seed, inclusive, nothing, nothing, items,
                subgroups, ndrange=block_size)
        return v
    end

    # The tiles' totals are scanned the same way, with the rest of `prefixes` for the totals'
    # totals (from `seed`, as the first total includes `start`)
    totals = view(prefixes, 1:num_blocks)
    scan_totals!() =
        accumulate_1d_gpu!(op, totals, backend, alg; init=NoInit(), seed, inclusive=true,
                           prefixes=view(prefixes, num_blocks+1:length(prefixes)))
    if scan_tuning(backend, eltype(v)).reduce_then_scan
        # Reduce the tiles, then scan them from their prefix: this reads the source twice and
        # writes the result once
        tile_aggregates_kernel!(backend, block_size)(op, totals, src, load, start, seed,
                                                     items, subgroups,
                                                     ndrange=num_blocks * block_size)
        scan_totals!()
        kernel!(op, v, src, load, start, seed, inclusive, nothing, totals, items,
                subgroups, ndrange=num_blocks * block_size)
    else
        # Scan the tiles, then add their prefix: this reads and writes the result twice
        kernel!(op, v, src, load, start, seed, inclusive, totals, nothing, items,
                subgroups, ndrange=num_blocks * block_size)
        scan_totals!()
        add_tile_prefixes_kernel!(backend, block_size)(op, v, totals, seed, inclusive, items,
                                                       ndrange=(num_blocks - 1) * block_size)
    end
    return v
end
