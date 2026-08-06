const ACC_NUM_BANKS::UInt8 = 32
const ACC_LOG_NUM_BANKS::UInt8 = 5

const ACC_FLAG_A::UInt8 = 0             # Aggregate of all previous prefixes finished
const ACC_FLAG_P::UInt8 = 1             # Only current block's prefix available


# Bank-conflict-avoiding padding, used by the multi-dimensional scan (accumulate_nd.jl)
@inline function conflict_free_offset(n)
    n >> ACC_LOG_NUM_BANKS
end


# ─── Block scan — register raking, ITEMS elements per thread ─────────────────
# Each block scans a tile of `block_size * ITEMS` elements.  Global loads/stores
# are striped (coalesced); each thread then serial-scans its own contiguous run
# of ITEMS elements in registers, the block_size run-totals are scanned once, and
# each thread folds its exclusive block-prefix back into its run.  Raising ITEMS
# shrinks the block count (and the per-(block) prefix array) without deepening the
# tree — the tree stays a single scan over `block_size` aggregates whatever ITEMS
# is, so a block covers more elements at the same barrier cost.  `prefixes[b]` gets
# the block's inclusive total; `flags` is only written for DecoupledLookback.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_block!(
    op, v, init, neutral,
    inclusive,
    flags, prefixes,                # one per block
    ::Val{ITEMS},
) where ITEMS
    # NOTE: block_size MUST be a power of 2
    @uniform block_size = @groupsize()[1]
    s   = @localmem eltype(v) (block_size * ITEMS,)   # block tile, natural order
    agg = @localmem eltype(v) (block_size,)           # per-thread run totals → scanned

    # NOTE: internal index maths are zero-based (fewer ops, matches the transpiled
    # CUDA/ROCm/oneAPI/Metal code); the `+ 0x1` appears only at memory accesses.
    len = length(v)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    # Phase 0: striped (coalesced) load into shared, natural order; OOB → neutral
    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        s[p + 0x1] = gi < len ? v[gi + 0x1] : neutral
        j += 1
    end
    @synchronize()

    # Phase 1: serial inclusive scan of this thread's run [t*ITEMS, t*ITEMS+ITEMS)
    run = ithread * ITEMS
    acc = neutral
    k = 0
    while k < ITEMS
        acc = op(acc, s[run + k + 0x1])
        s[run + k + 0x1] = acc          # keep run-local inclusive partials
        k += 1
    end
    agg[ithread + 0x1] = acc            # this thread's run total
    @synchronize()

    # Phase 2: exclusive scan of the block_size run-totals with a work-efficient
    # Blelloch tree — one barrier per level (the pattern POCL/SPIR-V lowers cleanly).
    # Block 0 seeds the running total with `init`; other blocks seed `neutral` and
    # get their real carry from kernel 2.
    seed = iblock == 0x0 ? init : neutral

    # Up-sweep (reduce).  Counters are index-typed (not UInt8) so they don't wrap
    # at block_size >= 256.
    offset = one(ithread)
    d = block_size >> 0x1
    while d > 0x0
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            agg[bi + 0x1] = op(agg[bi + 0x1], agg[ai + 0x1])
        end
        offset = offset << 0x1
        d = d >> 0x1
    end

    @synchronize()
    block_total = op(seed, agg[block_size - 0x1 + 0x1])   # inclusive total (root = full sum)
    @synchronize()
    if ithread == 0x0
        agg[block_size - 0x1 + 0x1] = seed                # clear root before down-sweep
    end

    # Down-sweep → exclusive scan, agg[t] = op(seed, sum of run-totals[0..t))
    d = one(ithread)
    while d < block_size
        offset = offset >> 0x1
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            t = agg[ai + 0x1]
            agg[ai + 0x1] = agg[bi + 0x1]
            agg[bi + 0x1] = op(agg[bi + 0x1], t)
        end
        d = d << 0x1
    end
    @synchronize()
    agg_excl = agg[ithread + 0x1]

    # Phase 3: fold the exclusive prefix into the run, writing the requested scan.
    # DecoupledLookback needs non-first blocks inclusive (kernel 2 adds a globally
    # inclusive carry); ScanPrefixes exclusive scans stay exclusive.  `do_incl` is
    # uniform across the group, so no barrier sits inside a divergent branch.
    do_incl = inclusive || (iblock != 0x0 && !isnothing(flags))
    prev = neutral                      # run-local inclusive at k-1 (neutral before the run)
    k = 0
    while k < ITEMS
        r = s[run + k + 0x1]            # run-local inclusive at k
        s[run + k + 0x1] = do_incl ? op(agg_excl, r) : op(agg_excl, prev)
        prev = r
        k += 1
    end
    @synchronize()

    # Block total and flag (block-uniform values, written once)
    if ithread == 0x0
        if !isnothing(prefixes)
            prefixes[iblock + 0x1] = block_total
        end
        if !isnothing(flags)
            flags[iblock + 0x1] = ACC_FLAG_P
        end
    end

    # Striped (coalesced) store, in-bounds only
    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        gi = block_offset + p
        if gi < len
            v[gi + 0x1] = s[p + 0x1]
        end
        j += 1
    end
end


# Add each block's running prefix (sum of all previous blocks) to its ITEMS-per-
# thread tile.  DecoupledLookback: walk back over predecessors, short-circuiting
# on the first block that has published its full aggregate.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous!(
    op, v, flags, @Const(prefixes), ::Val{ITEMS},
) where ITEMS
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock  = @index(Group, Linear) - 0x1 + 0x1              # Skipping first block
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    # Each block looks back to find running prefix sum
    running_prefix = prefixes[iblock - 0x1 + 0x1]
    inspected_block = signed(typeof(iblock))(iblock) - 0x2
    while inspected_block >= 0x0
        # Opportunistic: a previous block finished everything
        if UnsafeAtomics.load(pointer(flags, inspected_block + 0x1), UnsafeAtomics.monotonic) == ACC_FLAG_A
            UnsafeAtomics.fence(UnsafeAtomics.acquire) # (fence before reading from v)
            # Previous blocks (except last) always have filled values in v, so index is inbounds
            running_prefix = op(running_prefix, v[(inspected_block + 0x1) * block_size * ITEMS])
            break
        else
            running_prefix = op(running_prefix, prefixes[inspected_block + 0x1])
        end

        inspected_block -= 0x1
    end

    # Add the aggregate prefix of all previous blocks to each of our elements
    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = op(running_prefix, v[gi + 0x1])
        end
        j += 1
    end

    # Set flag for "aggregate of all prefixes up to this block finished".  The fence
    # + atomic store publish our writes to `v` before the flag becomes visible.
    @synchronize() # within-block
    UnsafeAtomics.fence(UnsafeAtomics.release)
    if ithread == 0x0
        UnsafeAtomics.store!(pointer(flags, iblock + 0x1), convert(eltype(flags), ACC_FLAG_A), UnsafeAtomics.monotonic)
    end
end


# ScanPrefixes carry: the prefixes were pre-scanned in chunks of `block_size*ITEMS`;
# fold in any earlier chunks, then add the running prefix to this block's tile.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous_coupled_preblocks!(
    op, v, prefixes, ::Val{ITEMS},
) where ITEMS
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock  = @index(Group, Linear) - 0x1 + 0x1              # Skipping first block
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    running_prefix = prefixes[iblock - 0x1 + 0x1]

    # If there were more than `block_size*ITEMS` prefixes, each chunk was scanned
    # internally but not across chunks; fold the earlier chunks' totals in here.
    num_preblocks = (iblock - 0x1) ÷ (block_size * ITEMS)
    for i in 0x1:num_preblocks
        running_prefix = op(running_prefix, prefixes[i * block_size * ITEMS])
    end

    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = op(running_prefix, v[gi + 0x1])
        end
        j += 1
    end
end


# DecoupledLookback algorithm
function accumulate_1d_gpu!(
    op, v::AbstractArray, backend::Backend, ::DecoupledLookback;
    init,
    neutral,
    inclusive::Bool,

    # CPU settings - not used
    max_tasks::Int,
    min_elems::Int,

    # GPU settings
    block_size::Int,
    items_per_thread::Int,
    temp::Union{Nothing, AbstractArray},
    temp_flags::Union{Nothing, AbstractArray},
)
    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)
    @argcheck items_per_thread > 0

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    elems_per_block = block_size * items_per_thread
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block
    valitems = Val(items_per_thread)

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = temp
    end

    if isnothing(temp_flags)
        flags = similar(v, UInt8, num_blocks)
    else
        @argcheck eltype(temp_flags) <: Integer
        @argcheck length(temp_flags) >= num_blocks
        flags = temp_flags
    end

    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, init, neutral, inclusive, flags, prefixes, valitems,
             ndrange=num_blocks * block_size)

    if num_blocks > 1
        kernel2! = _accumulate_previous!(backend, block_size)
        kernel2!(op, v, flags, prefixes, valitems,
                 ndrange=(num_blocks - 1) * block_size)
    end

    return v
end


# ScanPrefixes algorithm
function accumulate_1d_gpu!(
    op, v::AbstractArray, backend, ::ScanPrefixes;
    init,
    neutral,
    inclusive::Bool,

    # CPU settings - not used
    max_tasks::Int,
    min_elems::Int,

    # GPU settings
    block_size::Int,
    items_per_thread::Int,
    temp::Union{Nothing, AbstractArray},
    temp_flags::Union{Nothing, AbstractArray},
)
    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)
    @argcheck items_per_thread > 0

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    elems_per_block = block_size * items_per_thread
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block
    valitems = Val(items_per_thread)

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = temp
    end

    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, init, neutral, inclusive, nothing, prefixes, valitems,
             ndrange=num_blocks * block_size)

    if num_blocks > 1

        # Accumulate prefixes of all blocks; use neutral as init here to not reinclude init
        num_blocks_prefixes = (length(prefixes) + elems_per_block - 1) ÷ elems_per_block
        kernel1!(op, prefixes, neutral, neutral, true, nothing, nothing, valitems,
                 ndrange=num_blocks_prefixes * block_size)

        # Prefixes are pre-accumulated (completely accumulated if num_blocks_prefixes == 1, or
        # partially, which we will account for in the coupled lookback)
        kernel2! = _accumulate_previous_coupled_preblocks!(backend, block_size)
        kernel2!(op, v, prefixes, valitems,
                 ndrange=(num_blocks - 1) * block_size)
    end

    return v
end
