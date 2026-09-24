function accumulate_nd!(
    op, v::AbstractArray, backend::Backend, alg::Union{SliceScan, CPUThreads.Partitioned};
    init,
    neutral,
    dims::Int,
    inclusive::Bool,
)
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
        _accumulate_nd_cpu_sections!(op, v; seed, neutral, dims, inclusive,
                                     max_tasks=alg.max_tasks, min_elems=alg.min_elems)
    else
        block_size = alg.block_size
        # On GPUs we have two parallelisation approaches, based on which dimension has more elements:
        #   - If the dimension we are accumulating along has more elements than the "outer" dimensions,
        #     (e.g. accumulate(+, rand(3, 1000), dims=2)), we use a block of threads per outer
        #     dimension - thus, a block of threads reduces the dims axis
        #   - If the other dimensions have more elements (e.g. reduce(+, rand(3, 1000), dims=1)), we
        #     use a single thread per outer dimension - thus, a thread reduces the dims axis
        #     sequentially, while the other dimensions are processed in parallel, independently
        length_dims = vsizes[dims]
        length_outer = length(v) ÷ length_dims

        if length_outer >= length_dims
            # One thread per outer dimension
            blocks = (length_outer + block_size - 1) ÷ block_size
            kernel1! = _accumulate_nd_by_thread!(backend, block_size)
            kernel1!(
                v, op, seed, neutral, dims, inclusive,
                ndrange=(block_size * blocks,),
            )
        else
            # One block per outer dimension
            blocks = length_outer
            kernel2! = _accumulate_nd_by_block!(backend, block_size)
            kernel2!(
                v, op, seed, neutral, dims, inclusive,
                ndrange=(block_size, blocks),
            )
        end
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


@kernel inbounds=true cpu=false unsafe_indices=true function _accumulate_nd_by_thread!(
    v, op, seed, neutral, dims, inclusive,
)
    _, op = _lanefuncs(_Partials(), op, neutral)
    # One thread per outer dimension element, when there are more outer elements than in the
    # reduced dim e.g. accumulate(+, rand(3, 1000), dims=1) => only 3 elements in the accumulated
    # dim
    vsizes = size(v)
    vstrides = Base.size_to_strides(1, vsizes...)

    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    block_size = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each thread handles one outer element
    tid = ithread + iblock * block_size
    if tid < length_outer

        # Compute the base index in v for this thread
        input_base_idx = typeof(iblock)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in 0x1:ndims
            if i != dims
                input_base_idx += (tmp % vsizes[i]) * vstrides[i]
                tmp = tmp ÷ vsizes[i]
            end
        end

        # Go over each element in the accumulated dimension; this implementation assumes that there
        # are so many outer elements (each processed by an independent thread) that we afford to
        # loop sequentially over the accumulated dimension (e.g. reduce(+, rand(3, 1000), dims=1))
        running = seed
        for i in 0x0:length_dims - 0x1
            v_idx = input_base_idx + i * vstrides[dims]
            x = _lift(neutral, v[v_idx + 0x1])
            if inclusive
                running = op(running, x)
                v[v_idx + 0x1] = _unlane(running)
            else
                v[v_idx + 0x1] = _unlane(running)
                running = op(running, x)
            end
        end
    end
end


@kernel inbounds=true cpu=false unsafe_indices=true function _accumulate_nd_by_block!(
    v, op, seed, neutral, dims, inclusive,
)
    # NOTE: shmem_size MUST be greater than 2 * block_size
    # NOTE: block_size MUST be a power of 2

    # One block per outer dimension element, when there are more elements in the accumulated dim
    # than in outer dimensions, e.g. accumulate(+, rand(3, 1000), dims=2) => only 3 elements in
    # outer dimensions
    vsizes = size(v)
    vstrides = Base.size_to_strides(1, vsizes...)

    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    @uniform block_size = @groupsize()[1]

    temp = @localmem typeof(neutral) (0x2 * block_size + conflict_free_offset(0x2 * block_size),)
    running_prefix = @localmem typeof(neutral) (1,)
    _, op = _lanefuncs(_Partials(), op, neutral)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block handles one outer element; guaranteed to have exact number of blocks, so no need
    # for `if iblock < length_outer`

    # Compute the base index in v for this block (all threads in the block share the same)
    input_base_idx = typeof(iblock)(0)
    tmp = iblock
    KernelAbstractions.Extras.@unroll for i in 0x1:ndims
        if i != dims
            input_base_idx += (tmp % vsizes[i]) * vstrides[i]
            tmp = tmp ÷ vsizes[i]
        end
    end

    # We have a block of threads to accumulate along the dims axis; do it in chunks of
    # 2 * block_size and carry the total of all previous chunks (seeded with `seed`) into each one.
    # Operands are combined in element order, so `op` need not be commutative.
    ichunk = typeof(iblock)(0)
    num_chunks = (length_dims + (0x2 * block_size) - 0x1) ÷ (0x2 * block_size)

    if ithread == 0x0
        running_prefix[0x1] = seed
    end

    while ichunk < num_chunks
        block_offset = ichunk * block_size * 0x2            # Processing two elements per thread

        # Copy two elements from the main array; offset indices to avoid bank conflicts
        ai = ithread
        bi = ithread + block_size

        bank_offset_a = conflict_free_offset(ai)
        bank_offset_b = conflict_free_offset(bi)

        xa = if block_offset + ai < length_dims
            _lift(neutral, v[
                input_base_idx +                            # Outer element axis starting index
                (block_offset + ai) * vstrides[dims] +      # Move along dims axis in strides
                0x1                                         # - to 1-indexing
            ])
        else
            neutral
        end
        xb = if block_offset + bi < length_dims
            _lift(neutral, v[
                input_base_idx +
                (block_offset + bi) * vstrides[dims] +
                0x1
            ])
        else
            neutral
        end

        # The previous iteration's reads of `temp` finished before its last barrier
        temp[ai + bank_offset_a + 0x1] = xa
        temp[bi + bank_offset_b + 0x1] = xb

        # Build block reduction down
        offset = typeof(ithread)(1)
        next_pow2 = block_size * 0x2
        d = next_pow2 >> 0x1
        while d > 0x0             # TODO: unroll this like in reduce.jl ?
            @synchronize()

            if ithread < d
                _ai = offset * (0x2 * ithread + 0x1) - 0x1
                _bi = offset * (0x2 * ithread + 0x2) - 0x1
                _ai += conflict_free_offset(_ai)
                _bi += conflict_free_offset(_bi)

                temp[_bi + 0x1] = op(temp[_ai + 0x1], temp[_bi + 0x1])
            end

            offset = offset << 0x1
            d = d >> 0x1
        end

        # Flush last element
        if ithread == 0x0
            offset0 = conflict_free_offset(next_pow2 - 0x1)
            temp[next_pow2 - 0x1 + offset0 + 0x1] = neutral
        end

        # Build block accumulation up, giving an exclusive scan of the chunk
        d = typeof(ithread)(1)
        while d < next_pow2
            offset = offset >> 0x1
            @synchronize()

            if ithread < d
                _ai = offset * (0x2 * ithread + 0x1) - 0x1
                _bi = offset * (0x2 * ithread + 0x2) - 0x1
                _ai += conflict_free_offset(_ai)
                _bi += conflict_free_offset(_bi)

                t = temp[_ai + 0x1]
                temp[_ai + 0x1] = temp[_bi + 0x1]
                temp[_bi + 0x1] = op(temp[_bi + 0x1], t)
            end

            d = d << 0x1
        end
        @synchronize()

        # Exclusive prefixes within the chunk; include the element itself for inclusive scans
        ea = temp[ai + bank_offset_a + 0x1]
        eb = temp[bi + bank_offset_b + 0x1]
        ra = inclusive ? op(ea, xa) : ea
        rb = inclusive ? op(eb, xb) : eb
        carry = running_prefix[0x1]

        if block_offset + ai < length_dims
            v[
                input_base_idx +
                (block_offset + ai) * vstrides[dims] +
                0x1
            ] = _unlane(op(carry, ra))
        end
        if block_offset + bi < length_dims
            v[
                input_base_idx +
                (block_offset + bi) * vstrides[dims] +
                0x1
            ] = _unlane(op(carry, rb))
        end

        # Every thread has read the carry; the last thread extends it by this chunk's total (the
        # padding past the end of the slice is `neutral`)
        @synchronize()
        if bi == 0x2 * block_size - 0x1
            running_prefix[0x1] = op(carry, op(eb, xb))
        end

        ichunk += 0x1
    end
end
