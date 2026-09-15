const ACC_NUM_BANKS::UInt8 = 32
const ACC_LOG_NUM_BANKS::UInt8 = 5

const ACC_FLAG_A::UInt8 = 0             # Aggregate of all previous prefixes finished
const ACC_FLAG_P::UInt8 = 1             # Only current block's prefix available


@inline function conflict_free_offset(n)
    # Two possible offsets
    n >> ACC_LOG_NUM_BANKS
    # n >> ACC_NUM_BANKS + n >> (2 * ACC_LOG_NUM_BANKS)
end


@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_block!(
    op, v, init, neutral,
    inclusive,
    flags, prefixes,                # one per block
)
    # NOTE: shmem_size MUST be greater than 2 * block_size
    # NOTE: block_size MUST be a power of 2
    len = length(v)
    @uniform block_size = @groupsize()[1]
    temp = @localmem eltype(v) (typeof(block_size)(2) * block_size + conflict_free_offset(typeof(block_size)(2) * block_size),)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    num_blocks = Base.unsafe_trunc(typeof(block_size), @ndrange()[1]) ÷ block_size
    block_offset = iblock * block_size * typeof(iblock)(2)            # Processing two elements per thread

    # Copy two elements from the main array; offset indices to avoid bank conflicts
    ai = ithread
    bi = ithread + block_size

    bank_offset_a = conflict_free_offset(ai)
    bank_offset_b = conflict_free_offset(bi)

    if block_offset + ai < len
        @inbounds temp[ai + bank_offset_a + 0x1] = @inbounds v[block_offset + ai + 0x1]
    else
        @inbounds temp[ai + bank_offset_a + 0x1] = neutral
    end

    if block_offset + bi < len
        @inbounds temp[bi + bank_offset_b + 0x1] = @inbounds v[block_offset + bi + 0x1]
    else
        @inbounds temp[bi + bank_offset_b + 0x1] = neutral
    end

    # Build block reduction down
    offset = typeof(ithread)(1)
    next_pow2 = block_size * typeof(block_size)(2)
    d = next_pow2 >> 0x1
    while d > typeof(d)(0)             # TODO: unroll this like in reduce.jl ?
        @synchronize()

        if ithread < d
            _ai = offset * (typeof(ithread)(2) * ithread + typeof(ithread)(1)) - typeof(ithread)(1)
            _bi = offset * (typeof(ithread)(2) * ithread + typeof(ithread)(2)) - typeof(ithread)(1)
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)

            @inbounds temp[_bi + 0x1] = op(@inbounds(temp[_bi + 0x1]), @inbounds(temp[_ai + 0x1]))
        end

        offset = offset << 0x1
        d = d >> 0x1
    end

    # Flush last element
    if ithread == typeof(ithread)(0)
        offset0 = conflict_free_offset(next_pow2 - typeof(next_pow2)(1))
        @inbounds temp[(next_pow2 - typeof(next_pow2)(1) + offset0 + typeof(next_pow2)(1))] = iblock == 0x0 ? init : neutral
    end

    # Build block accumulation up
    d = typeof(ithread)(1)
    while d < next_pow2
        offset = offset >> 0x1
        @synchronize()

        if ithread < d
            _ai = offset * (typeof(ithread)(2) * ithread + typeof(ithread)(1)) - typeof(ithread)(1)
            _bi = offset * (typeof(ithread)(2) * ithread + typeof(ithread)(2)) - typeof(ithread)(1)
            _ai += conflict_free_offset(_ai)
            _bi += conflict_free_offset(_bi)

            t = @inbounds temp[_ai + 0x1]
            @inbounds temp[_ai + 0x1] = @inbounds temp[_bi + 0x1]
            @inbounds temp[_bi + 0x1] = op(@inbounds(temp[_bi + 0x1]), t)
        end

        d = d << 0x1
    end

    # Later blocks should always be inclusively-scanned
    if inclusive || iblock != typeof(iblock)(0)
        # To compute an inclusive scan, shift elements left...
        @synchronize()
        t1 = @inbounds(temp[ai + bank_offset_a + 0x1])
        t2 = @inbounds(temp[bi + bank_offset_b + 0x1])
        @synchronize()

        if ai > typeof(ai)(0)
            @inbounds temp[ai - 0x1 + conflict_free_offset(ai - 0x1) + 0x1] = t1
        end
        @inbounds temp[bi - 0x1 + conflict_free_offset(bi - 0x1) + 0x1] = t2

        # ...and accumulate the last value too
        if bi == typeof(bi)(2) * block_size - typeof(bi)(1)
            if iblock < num_blocks - typeof(iblock)(1)
                @inbounds temp[bi + bank_offset_b + 0x1] = op(t2, @inbounds(v[(iblock + 0x1) * block_size * 0x2]))
            else
                @inbounds temp[bi + bank_offset_b + 0x1] = op(t2, @inbounds(v[len]))
            end
        end
    end

    @synchronize()

    # Write this block's final prefix to global array and set flag to "block prefix computed"
    if bi == typeof(bi)(2) * block_size - typeof(bi)(1)

        # Known at compile-time; used in the first pass of the ScanPrefixes algorithm
        if !isnothing(prefixes)
            @inbounds prefixes[iblock + 0x1] = @inbounds(temp[bi + bank_offset_b + 0x1])
        end

        # Known at compile-time; used only in the DecoupledLookback algorithm
        if !isnothing(flags)
            @inbounds flags[iblock + 0x1] = ACC_FLAG_P
        end
    end

    if block_offset + ai < len
        @inbounds v[block_offset + ai + 0x1] = @inbounds(temp[ai + bank_offset_a + 0x1])
    end
    if block_offset + bi < len
        @inbounds v[block_offset + bi + 0x1] = @inbounds(temp[bi + bank_offset_b + 0x1])
    end
end


@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous!(
    op, v, flags, @Const(prefixes),
)

    len = length(v)
    block_size = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1 + 0x1              # Skipping first block
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * typeof(iblock)(2)                # Processing two elements per thread

    # Each block looks back to find running prefix sum
    running_prefix = @inbounds(prefixes[iblock - 0x1 + 0x1])
    # Use bitcast for UInt32→Int32 to avoid checked conversion (check_sign_bit) on GPU.
    # Safe because iblock is always a small positive block index (high bit never set).
    inspected_block = Base.bitcast(Int32, iblock) - Int32(2)
    while inspected_block >= Int32(0)
        # Opportunistic: a previous block finished everything
        if UnsafeAtomics.load(pointer(flags, inspected_block + Int32(1)), UnsafeAtomics.monotonic) == ACC_FLAG_A
            UnsafeAtomics.fence(UnsafeAtomics.acquire) # (fence before reading from v)
            # Previous blocks (except last) always have filled values in v, so index is inbounds
            running_prefix = op(running_prefix, @inbounds(v[(inspected_block + Int32(1)) * block_size * typeof(block_size)(2)]))
            break
        else
            running_prefix = op(running_prefix, @inbounds(prefixes[inspected_block + 0x1]))
        end

        inspected_block -= Int32(1)
    end

    # Now we have aggregate prefix of all previous blocks, add it to all our elements
    ai = ithread
    if block_offset + ai < len
        @inbounds v[block_offset + ai + 0x1] = op(running_prefix, @inbounds(v[block_offset + ai + 0x1]))
    end

    bi = ithread + block_size
    if block_offset + bi < len
        @inbounds v[block_offset + bi + 0x1] = op(running_prefix, @inbounds(v[block_offset + bi + 0x1]))
    end

    # Set flag for "aggregate of all prefixes up to this block finished"
    # There are two synchronization concerns here:
    # 1. Withing a group we want to ensure that all writed to `v` have occured before setting the flag.
    # 2. Between groups we need to use a fence and atomic load/store to ensure that memory operations are not re-ordered
    @synchronize() # within-block
    # Note: This fence is needed to ensure that the flag is not set before copying into v.
    #       See https://doc.rust-lang.org/std/sync/atomic/fn.fence.html
    #       for more details.
    #       We use the happens-before relation between stores to `v` and the store to `flags`.
    UnsafeAtomics.fence(UnsafeAtomics.release)
    if ithread == typeof(ithread)(0)
        UnsafeAtomics.store!(pointer(flags, iblock + typeof(iblock)(1)), convert(eltype(flags), ACC_FLAG_A), UnsafeAtomics.monotonic)
    end
end


@kernel cpu=false inbounds=true unsafe_indices=true function _accumulate_previous_coupled_preblocks!(
    op, v, prefixes,
)
    # No decoupled lookback
    len = length(v)
    block_size = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1 + 0x1              # Skipping first block
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * typeof(iblock)(2)                # Processing two elements per thread

    # Each block looks back to find running prefix sum
    running_prefix = @inbounds(prefixes[iblock - 0x1 + 0x1])

    # The prefixes were pre-accumulated, which means (for block_size=N):
    #   - If there were N or fewer prefixes (so fewer than N*N elements in v to begin with), the
    #     prefixes were fully accumulated and we can use them directly.
    #   - If there were more than N prefixes, each chunk of N prefixes was accumulated, but not
    #     along the chunks. We need to accumulate the prefixes of the previous chunks into
    #     running_prefix.
    num_preblocks = (iblock - typeof(iblock)(1)) ÷ (block_size * typeof(block_size)(2))
    i = typeof(iblock)(1)
    while i <= num_preblocks
        running_prefix = op(running_prefix, @inbounds(prefixes[i * block_size * typeof(block_size)(2)]))
        i += typeof(i)(1)
    end

    # Now we have aggregate prefix of all previous blocks, add it to all our elements
    ai = ithread
    if block_offset + ai < len
        @inbounds v[block_offset + ai + 0x1] = op(running_prefix, @inbounds(v[block_offset + ai + 0x1]))
    end

    bi = ithread + block_size
    if block_offset + bi < len
        @inbounds v[block_offset + bi + 0x1] = op(running_prefix, @inbounds(v[block_offset + bi + 0x1]))
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
    temp::Union{Nothing, AbstractArray},
    temp_flags::Union{Nothing, AbstractArray},
)
    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    # Each thread will process two elements
    elems_per_block = block_size * 2
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block

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
    init_typed = convert(eltype(v), init)
    kernel1!(op, v, init_typed, neutral, inclusive, flags, prefixes,
             ndrange=num_blocks * block_size)

    if num_blocks > 1
        kernel2! = _accumulate_previous!(backend, block_size)
        kernel2!(op, v, flags, prefixes,
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
    temp::Union{Nothing, AbstractArray},
    temp_flags::Union{Nothing, AbstractArray},
)
    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)

    # Nothing to accumulate
    if length(v) == 0
        return v
    end

    # Each thread will process two elements
    elems_per_block = block_size * 2
    num_blocks = (length(v) + elems_per_block - 1) ÷ elems_per_block

    if isnothing(temp)
        prefixes = similar(v, eltype(v), num_blocks)
    else
        @argcheck eltype(temp) === eltype(v)
        @argcheck length(temp) >= num_blocks
        prefixes = temp
    end

    kernel1! = _accumulate_block!(backend, block_size)
    init_typed = convert(eltype(v), init)
    kernel1!(op, v, init_typed, neutral, inclusive, nothing, prefixes,
             ndrange=num_blocks * block_size)

    if num_blocks > 1

        # Accumulate prefixes of all blocks; use neutral as init here to not reinclude init
        num_blocks_prefixes = (length(prefixes) + elems_per_block - 1) ÷ elems_per_block
        kernel1!(op, prefixes, neutral, neutral, true, nothing, nothing,
                 ndrange=num_blocks_prefixes * block_size)

        # Prefixes are pre-accumulated (completely accumulated if num_blocks_prefixes == 1, or
        # partially, which we will account for in the coupled lookback)
        kernel2! = _accumulate_previous_coupled_preblocks!(backend, block_size)
        kernel2!(op, v, prefixes,
                 ndrange=(num_blocks - 1) * block_size)
    end

    return v
end
