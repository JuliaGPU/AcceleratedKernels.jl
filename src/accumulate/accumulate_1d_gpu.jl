const ACC_LOG_NUM_BANKS::UInt8 = 5

const ACC_FLAG_A::UInt8 = 0             # Aggregate of all previous prefixes finished
const ACC_FLAG_P::UInt8 = 1             # Only current block's prefix available


# Bank-conflict-avoiding padding, used by the multi-dimensional scan (accumulate_nd.jl)
@inline function conflict_free_offset(n)
    n >> ACC_LOG_NUM_BANKS
end


# Device-scope memory fence for the DecoupledLookback scan. Each GPU backend overrides it with a
# native device fence in its package extension; a plain UnsafeAtomics.fence is not device scoped.
function _decoupled_fence end


# Exclusive scan of one value per thread in local memory. All threads in the block must call it.
@inline function block_exclusive_scan!(@context, op, totals, seed, block_size, ithread)
    # Up-sweep. Use index-sized counters for block sizes of 256 or more.
    offset = one(ithread)
    d = block_size >> 0x1
    while d > 0x0
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            totals[bi + 0x1] = op(totals[bi + 0x1], totals[ai + 0x1])
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


# Register-raking block scan with striped loads and stores.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_block!(
    op, v, init, neutral,
    inclusive,
    flags, prefixes,
    ::Val{ITEMS},
) where ITEMS
    # `block_size` is a power of two.
    @uniform block_size = @groupsize()[1]
    tile = @localmem eltype(v) (block_size * ITEMS,)
    thread_totals = @localmem eltype(v) (block_size,)

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
        tile[p + 0x1] = gi < len ? v[gi + 0x1] : neutral
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

    # Scan the per-thread totals. Later blocks receive their carry from the
    # second kernel.
    seed = iblock == 0x0 ? init : neutral
    thread_prefix, block_total = block_exclusive_scan!(
        @context, op, thread_totals, seed, block_size, ithread,
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
            v[gi + 0x1] = tile[p + 0x1]
        end
        j += 1
    end
end


# Add each block's running prefix, stopping at a completed predecessor.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous!(
    op, v, flags, @Const(prefixes), ::Val{ITEMS},
) where ITEMS
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    running_prefix = prefixes[iblock]
    inspected_block = signed(typeof(iblock))(iblock) - 0x2
    while inspected_block >= 0x0
        flag = UnsafeAtomics.load(
            pointer(flags, inspected_block + 0x1),
            UnsafeAtomics.monotonic,
        )
        if flag == ACC_FLAG_A
            _decoupled_fence()          # acquire: order the `v` read after the flag load
            running_prefix = op(running_prefix, v[(inspected_block + 0x1) * block_size * ITEMS])
            break
        else
            running_prefix = op(running_prefix, prefixes[inspected_block + 0x1])
        end

        inspected_block -= 0x1
    end

    # Add the aggregate prefix of all previous blocks to each element.
    j = 0
    while j < ITEMS
        gi = block_offset + j * block_size + ithread
        if gi < len
            v[gi + 0x1] = op(running_prefix, v[gi + 0x1])
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


# Add pre-scanned block prefixes to each tile.
@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous_coupled_preblocks!(
    op, v, prefixes, ::Val{ITEMS},
) where ITEMS
    len = length(v)
    @uniform block_size = @groupsize()[1]

    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    running_prefix = prefixes[iblock]

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


# Save the value preceding each tile before shifting the array in place.
@kernel cpu=false inbounds=true function exclusive_prefixes_kernel!(
    prefixes, @Const(v), init, elems_per_block,
)
    iblock = @index(Global, Linear) - 0x1
    prefixes[iblock + 0x1] = iblock == 0x0 ? init : v[iblock * elems_per_block]
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
            v[gi + 0x1] = p == 0x0 ? prefixes[iblock + 0x1] : tile[p]
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
    items = Val(items_per_thread)

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = view(temp, 1:num_blocks)
    end

    if isnothing(temp_flags)
        flags = similar(v, UInt8, num_blocks)
    else
        @argcheck eltype(temp_flags) <: Integer
        @argcheck length(temp_flags) >= num_blocks
        flags = view(temp_flags, 1:num_blocks)
    end

    shift_to_exclusive = !inclusive && num_blocks > 1
    block_inclusive = inclusive || shift_to_exclusive

    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, init, neutral, block_inclusive, flags, prefixes, items,
             ndrange=num_blocks * block_size)

    if num_blocks > 1
        kernel2! = _accumulate_previous!(backend, block_size)
        kernel2!(op, v, flags, prefixes, items,
                 ndrange=(num_blocks - 1) * block_size)
    end

    if shift_to_exclusive
        exclusive_prefixes_kernel!(backend, block_size)(prefixes, v, init, elems_per_block,
                                                        ndrange=num_blocks)
        exclusive_shift_kernel!(backend, block_size)(v, prefixes, items,
                                                     ndrange=num_blocks * block_size)
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
    items = Val(items_per_thread)

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = view(temp, 1:num_blocks)
    end

    kernel1! = _accumulate_block!(backend, block_size)
    kernel1!(op, v, init, neutral, inclusive, nothing, prefixes, items,
             ndrange=num_blocks * block_size)

    if num_blocks > 1

        # Accumulate prefixes of all blocks; use neutral as init here to not reinclude init
        num_blocks_prefixes = (length(prefixes) + elems_per_block - 1) ÷ elems_per_block
        kernel1!(op, prefixes, neutral, neutral, true, nothing, nothing, items,
                 ndrange=num_blocks_prefixes * block_size)

        # Prefixes are pre-accumulated (completely accumulated if num_blocks_prefixes == 1, or
        # partially, which we will account for in the coupled lookback)
        kernel2! = _accumulate_previous_coupled_preblocks!(backend, block_size)
        kernel2!(op, v, prefixes, items,
                 ndrange=(num_blocks - 1) * block_size)
    end

    return v
end
