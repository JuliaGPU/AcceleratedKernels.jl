
const ACC_FLAG_A::UInt8 = 0             # Aggregate of all previous prefixes finished
const ACC_FLAG_P::UInt8 = 1             # Only current block's prefix available


# Device-scope memory fence for the DecoupledLookback scan. Each GPU backend overrides it with a
# native device fence in its package extension; a plain UnsafeAtomics.fence is not device scoped.
function _decoupled_fence end


# The sub-group width of the scans of values of type `T` in blocks of `block_size` threads on
# `backend`, as a `Val` for the kernels: the backend's, where the tuning asks for sub-groups and
# they are consecutive threads (from KernelAbstractions 0.10), shuffle `T`, and fit a block
# between one and as many times as their width (the sub-group totals are scanned by a
# sub-group); else 0, to scan in local memory.
@static if isdefined(KernelAbstractions, :KernelInterface)
    function _scan_subgroups(backend, ::Type{T}, block_size) where {T}
        scan_tuning(backend, T).subgroups || return Val(0)
        KI.supports_linear_subgroups(backend) && KI.supports_shuffle(backend, T) || return Val(0)
        W = KI.sub_group_size(backend)
        return ispow2(W) && W <= block_size <= W * W ? Val(W) : Val(0)
    end
else
    _scan_subgroups(backend, T, block_size) = Val(0)
end


# Exclusive scan of one value per thread, `totals[ithread + 1]`, seeded with `seed`: returns the
# thread's prefix and the block's total. All threads in the block must call it. Operands are
# combined in element order (earlier on the left), so `op` need not be commutative. With a
# sub-group width `W` (see `_scan_subgroups`), it scans with sub-group operations, else in local
# memory. It overwrites `totals`.
@inline block_exclusive_scan!(@context, op, totals, seed, block_size, ithread, ::Val{0}) =
    block_exclusive_scan!(@context, op, totals, seed, block_size, ithread)

@inline function block_exclusive_scan!(@context, op, totals, seed, block_size, ithread,
                                       ::Val{W}) where {W}
    # scan each sub-group's values, then the sub-groups' totals, in one sub-group
    x = totals[ithread + 0x1]
    lane = ithread % W
    sg = ithread ÷ W
    nsg = block_size ÷ W
    incl = KI.sub_group_scan(op, x)
    @synchronize()
    if lane == W - 1
        totals[sg + 0x1] = incl
    end
    @synchronize()
    # (lanes beyond the sub-groups only affect results that are not used)
    t = lane < nsg ? totals[lane + 0x1] : seed
    sg_excl = KI.sub_group_exclusive_scan(op, t, seed)
    sg_prefix = KI.shfl(sg_excl, sg + 0x1)
    block_total = KI.shfl(op(sg_excl, t), nsg)
    prev = KI.shfl_up(incl, 1)
    @synchronize()
    return (lane == 0 ? sg_prefix : op(sg_prefix, prev)), block_total
end

# The scan in local memory
@inline function block_exclusive_scan!(@context, op, totals, seed, block_size, ithread)
    # Up-sweep. Use index-sized counters for block sizes of 256 or more.
    offset = one(ithread)
    d = block_size >> 0x1
    while d > 0x0
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            totals[bi + 0x1] = op(totals[ai + 0x1], totals[bi + 0x1])
        end
        offset = offset << 0x1
        d = d >> 0x1
    end

    @synchronize()
    block_total = op(seed, totals[block_size])
    @synchronize()
    if ithread == 0x0
        totals[block_size] = seed
    end

    # Down-sweep to an exclusive scan.
    d = one(ithread)
    while d < block_size
        offset = offset >> 0x1
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            t = totals[ai + 0x1]
            totals[ai + 0x1] = totals[bi + 0x1]
            totals[bi + 0x1] = op(totals[bi + 0x1], t)
        end
        d = d << 0x1
    end
    @synchronize()

    return totals[ithread + 0x1], block_total
end


# Register-raking block scan with striped loads and stores, of the elements `entry(src[i])` into
# `v` (`src` may be `v`). `neutral` seeds partial results (an empty `_Lane` when `op` has no known
# neutral element), and `seed` is the first block's. With `block_seeds`, the inclusive scan of
# the blocks' totals, block `b` is seeded with `block_seeds[b]`, the reduction of the blocks
# before it; else with `neutral`, to receive its carry from another kernel.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_block!(
    op, v, src, entry, seed, neutral,
    inclusive,
    flags, prefixes, block_seeds,
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

    # DecoupledLookback keeps later blocks inclusive until the carry pass.
    block_inclusive = inclusive || (iblock != 0x0 && !isnothing(flags))
    prev = neutral
    k = 0
    while k < ITEMS
        r = tile[run + k + 0x1]
        tile[run + k + 0x1] =
            block_inclusive ? op(thread_prefix, r) : op(thread_prefix, prev)
        prev = r
        k += 1
    end
    @synchronize()

    if ithread == 0x0
        if !isnothing(prefixes)
            prefixes[iblock + 0x1] = block_total
        end
        if !isnothing(flags)
            flags[iblock + 0x1] = ACC_FLAG_P
        end
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

# Add each block's running prefix, stopping at a completed predecessor.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous!(
    op, v, flags, @Const(prefixes), neutral, ::Val{ITEMS},
) where ITEMS
    _, op = _lanefuncs(_Partials(), op, neutral)
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    # The lookback walks towards earlier blocks, so each earlier aggregate is combined on the left.
    running_prefix = prefixes[iblock]
    inspected_block = signed(typeof(iblock))(iblock) - 0x2
    while inspected_block >= 0x0
        flag = UnsafeAtomics.load(
            pointer(flags, inspected_block + 0x1),
            UnsafeAtomics.monotonic,
        )
        if flag == ACC_FLAG_A
            _decoupled_fence()          # acquire: order the `v` read after the flag load
            running_prefix = op(_lift(neutral, v[(inspected_block + 0x1) * block_size * ITEMS]),
                                running_prefix)
            break
        else
            running_prefix = op(prefixes[inspected_block + 0x1], running_prefix)
        end

        inspected_block -= 0x1
    end

    # Add the aggregate prefix of all previous blocks to each element.
    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = _lower(eltype(v), op(running_prefix, _lift(neutral, v[gi + 0x1])))
        end
        j += 1
    end

    # Publish writes to `v` before marking the block complete.
    @synchronize()
    _decoupled_fence()                  # release: order the flag store after the `v` writes
    if ithread == 0x0
        UnsafeAtomics.store!(
            pointer(flags, iblock + 0x1),
            convert(eltype(flags), ACC_FLAG_A),
            UnsafeAtomics.monotonic,
        )
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

# Save the value preceding each tile before shifting the array in place.
@kernel cpu=false inbounds=true function exclusive_prefixes_kernel!(
    prefixes, @Const(v), seed, neutral, elems_per_block,
)
    iblock = @index(Global, Linear) - 0x1
    prefixes[iblock + 0x1] = iblock == 0x0 ? seed : _lift(neutral, v[iblock * elems_per_block])
end


@kernel cpu=false inbounds=true unsafe_indices=true function exclusive_shift_kernel!(
    v, @Const(prefixes), ::Val{ITEMS},
) where ITEMS
    @uniform block_size = @groupsize()[1]
    tile = @localmem eltype(v) (block_size * ITEMS,)

    len = length(v)
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        if gi < len
            tile[p + 0x1] = v[gi + 0x1]
        end
        j += 1
    end
    @synchronize()

    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        if gi < len
            v[gi + 0x1] = p == 0x0 ? _lower(eltype(v), prefixes[iblock + 0x1]) : tile[p]
        end
        j += 1
    end
end


# The first block's seed: `init` in the element type, else the neutral seed
_scan_first_seed(v, init, neutral) = _lift(neutral, convert(eltype(v), init))
_scan_first_seed(v, ::_NoInit, neutral) = neutral


# DecoupledLookback algorithm
function accumulate_1d_gpu!(
    op, v::AbstractArray, backend::Backend, alg::DecoupledLookback;
    src=v,
    entry=identity,
    init,
    neutral,
    inclusive::Bool,
    prefixes::Union{Nothing, AbstractArray},
    flags::Union{Nothing, AbstractArray},
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

    shift_to_exclusive = !inclusive && num_blocks > 1
    block_inclusive = inclusive || shift_to_exclusive

    subgroups = _scan_subgroups(backend, typeof(neutral), block_size)
    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, src, entry, seed, neutral, block_inclusive, flags, prefixes, nothing, items,
             subgroups, ndrange=num_blocks * block_size)

    if num_blocks > 1
        kernel2! = _accumulate_previous!(backend, block_size)
        kernel2!(op, v, flags, prefixes, neutral, items,
                 ndrange=(num_blocks - 1) * block_size)
    end

    if shift_to_exclusive
        exclusive_prefixes_kernel!(backend, block_size)(prefixes, v, seed, neutral,
                                                        elems_per_block, ndrange=num_blocks)
        exclusive_shift_kernel!(backend, block_size)(v, prefixes, items,
                                                     ndrange=num_blocks * block_size)
    end

    return v
end


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
    flags::Union{Nothing, AbstractArray},
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
        kernel!(op, v, src, entry, seed, neutral, inclusive, nothing, nothing, nothing, items,
                subgroups, ndrange=block_size)
        return v
    end

    # The tiles' totals are scanned the same way, with the rest of `prefixes` for the totals'
    # totals (from `neutral`, as the first total includes `seed`)
    totals = view(prefixes, 1:num_blocks)
    scan_totals!() =
        accumulate_1d_gpu!(op, totals, backend, alg; init=_NoInit(), neutral, inclusive=true,
                           prefixes=view(prefixes, num_blocks+1:length(prefixes)), flags=nothing)
    if scan_tuning(backend, eltype(v)).reduce_then_scan
        # Reduce the tiles, then scan them from their prefix: this reads the source twice and
        # writes the result once
        _accumulate_tile_totals!(backend, block_size)(op, totals, src, entry, seed, neutral,
                                                      items, subgroups,
                                                      ndrange=num_blocks * block_size)
        scan_totals!()
        kernel!(op, v, src, entry, seed, neutral, inclusive, nothing, nothing, totals, items,
                subgroups, ndrange=num_blocks * block_size)
    else
        # Scan the tiles, then add their prefix: this reads and writes the result twice
        kernel!(op, v, src, entry, seed, neutral, inclusive, nothing, totals, nothing, items,
                subgroups, ndrange=num_blocks * block_size)
        scan_totals!()
        _accumulate_add_prefixes!(backend, block_size)(op, v, totals, neutral, inclusive, items,
                                                       ndrange=(num_blocks - 1) * block_size)
    end
    return v
end
