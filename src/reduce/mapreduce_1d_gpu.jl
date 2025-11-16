function _mapreduce_block!(src, dst, f, op, neutral, ::Val{N}) where N
    @inbounds begin
    sdata = KI.localmemory(eltype(dst), N)
    N_actual = KI.get_local_size().x

    len = length(src)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = KI.get_group_id().x - 0x1
    ithread = KI.get_local_id().x - 0x1

    i = ithread + iblock * (N_actual * 0x2)
    if i >= len
        sdata[ithread + 0x1] = neutral
    elseif i + N_actual >= len
        sdata[ithread + 0x1] = f(src[i + 0x1])
    else
        sdata[ithread + 0x1] = op(f(src[i + 0x1]), f(src[i + N_actual + 0x1]))
    end

    KI.barrier()

    @inline reduce_group!(op, sdata, N_actual, ithread)

    # Code below would work on NVidia GPUs with warp size of 32, but create race conditions and
    # return incorrect results on Intel Graphics. It would be useful to have a way to statically
    # query the warp size at compile time
    #
    # if ithread < 32
    #     N >= 64 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 32 + 1]))
    #     N >= 32 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 16 + 1]))
    #     N >= 16 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 8 + 1]))
    #     N >= 8 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 4 + 1]))
    #     N >= 4 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 2 + 1]))
    #     N >= 2 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 1 + 1]))
    # end

    if ithread == 0x0
        dst[iblock + 0x1] = sdata[0x1]
    end
    end
    nothing
end


function mapreduce_1d_gpu(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral,

    # CPU settings - ignored here
    max_tasks::Int,
    min_elems::Int,

    # GPU settings
    block_size::Union{Nothing, Int},
    temp::Union{Nothing, AbstractArray},
    switch_below::Int,
)
    min_block_size = 16
    max_block_size = min(1024, get_max_block_size(backend, block_size))
    @argcheck 1 <= max_block_size <= 1024
    @argcheck switch_below >= 0

    # Degenerate cases
    len = length(src)
    len == 0 && return init
    len == 1 && return @allowscalar f(src[1])
    if len < switch_below
        h_src = Vector(src)
        return Base.mapreduce(f, op, h_src; init)
    end

    # Each thread will handle two elements
    # max_num_per_block = 2 * max_block_size
    min_num_per_block = 2 * min_block_size
    max_blocks = (len + min_num_per_block - 1) ÷ min_num_per_block

    if !isnothing(temp)
        @argcheck get_backend(temp) === backend
        @argcheck eltype(temp) === typeof(init)
        @argcheck length(temp) >= max_blocks * 2
        dst = temp
    else
        # Figure out type for destination
        dst_type = typeof(init)
        dst = KernelAbstractions.allocate(backend, dst_type, max_blocks * 2)
    end

    # Later the kernel will be compiled for views anyways, so use same types
    src_view = @view src[1:end]
    dst_view = @view dst[1:max_blocks]

    kernel = KI.@kernel backend launch = false _mapreduce_block!(src_view, dst_view, f, op, neutral, Val(max_block_size))

    workgroupsize = block_size_pow_2(kernel, block_size)
    numworkgroups = (len + workgroupsize - 1) ÷ workgroupsize

    dst_view = @view dst[1:numworkgroups]

    kernel(src_view, dst_view, f, op, neutral, Val(max_block_size); numworkgroups, workgroupsize)

    # As long as we still have blocks to process, swap between the src and dst pointers at
    # the beginning of the first and second halves of dst
    len = numworkgroups
    if len < switch_below
        h_src = Vector(@view(dst[1:len]))
        return Base.reduce(op, h_src; init)
    end

    # Now all src elements have been passed through f; just do final reduction, no map needed
    p1 = @view dst[1:len]
    p2 = @view dst[numworkgroups + 1:end]

    while len > 1
        kernel = KI.@kernel backend launch = false _mapreduce_block!(p1, p2, identity, op, neutral, Val(max_block_size))

        workgroupsize = block_size_pow_2(kernel, block_size)
        numworkgroups = (len + workgroupsize - 1) ÷ workgroupsize

        kernel(p1, p2, identity, op, neutral, Val(max_block_size); numworkgroups, workgroupsize)

        # Each block produces one reduced value
        len = numworkgroups

        if len < switch_below
            h_src = Vector(@view(p2[1:len]))
            return Base.reduce(op, h_src; init)
        end

        p1, p2 = p2, p1
        p1 = @view p1[1:len]
    end

    # The GPU kernel reduced all elements to one, but without the init value
    return op(init, @allowscalar(p1[1]))
end
