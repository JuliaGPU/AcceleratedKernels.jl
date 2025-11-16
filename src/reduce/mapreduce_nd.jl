function mapreduce_nd(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(src)),
    dims::Int,

    # CPU settings - ignored here
    max_tasks::Int,
    min_elems::Int,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Union{Nothing, Int},
    temp::Union{Nothing, AbstractArray},
)

    max_block_size = min(1024, get_max_block_size(backend, block_size))
    @argcheck 1 <= max_block_size <= 1024

    # Degenerate cases begin; order of priority matters

    # Invalid dims
    if dims < 1
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # If dims > number of dimensions, just map each element through f and add init, e.g.:
    #   julia> x = rand(Float64, 3, 5);
    #   julia> mapreduce(x -> -x, +, x, dims=3, init=Float32(0))
    #   3×5 Matrix{Float32}     # Negative numbers
    src_sizes = size(src)
    if dims > length(src_sizes)
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), src_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i == dims ? 1 : src_sizes[i]
    end

    # If any dimension except dims is zero, return empty similar array except with the dims
    # dimension = 1. Weird, see example below:
    #   julia> x = rand(3, 0, 5);
    #   julia> reduce(+, x, dims=3)
    #   3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize == dims && continue
        if src_sizes[isize] == 0
            if isnothing(temp)
                dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
            else
                @argcheck size(temp) == dst_sizes
                @argcheck eltype(temp) == typeof(init)
                dst = temp
            end
            return dst
        end
    end

    # If sizes[dims] == 0, return array filled with init; same shape except sizes[dims] = 1:
    #   julia> x = rand(3, 0, 5);
    #   julia> mapreduce(+, x, dims=2)
    #   3×1×5 Array{Float64, 3}:
    #   [:, :, 1] =
    #    0.0
    #    0.0
    #    0.0
    #   [...]
    len = src_sizes[dims]
    if len == 0
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == dst_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        fill!(dst, init)
        return dst
    end

    # If sizes[dims] == 1, just map each element through f. Again, keep same type as init
    if len == 1
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), src_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # Degenerate cases end

    # Allocate destination array
    if isnothing(temp)
        dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
    else
        @argcheck get_backend(temp) == backend
        @argcheck size(temp) == dst_sizes
        @argcheck eltype(temp) == typeof(init)
        dst = temp
    end
    dst_size = length(dst)

    if !use_gpu_algorithm(backend, prefer_threads)
        _mapreduce_nd_cpu_sections!(
            f, op, dst, src;
            init,
            dims,
            max_tasks=max_tasks,
            min_elems=min_elems,
        )
    else
        # On GPUs we have two parallelisation approaches, based on destination dimension and current hardware:
        #   - If the other dimensions have more elements than the product of the device's compute units and
        #     maximum number of threads , we use a single thread per dst element - thus, a thread reduces
        #     the dims axis sequentially, while the other dimensions are processed in parallel, independently
        #   - If the dimension we are reducing has more elements, we use a block of threads per dst
        #     element - thus, a block of threads reduces the dims axis
        by_thread_threshold = KI.max_work_group_size(backend) * KI.multiprocessor_count(backend)

        if dst_size >= by_thread_threshold
            kernel = KI.@kernel backend launch = false _mapreduce_nd_by_thread!(
                src, dst, f, op, init, dims
            )
            workgroupsize = isnothing(block_size) ? KI.kernel_max_work_group_size(kernel) : block_size
            numworkgroups = (dst_size + workgroupsize - 1) ÷ workgroupsize
            kernel(
                src, dst, f, op, init, dims; workgroupsize, numworkgroups
            )
        else
            kernel = KI.@kernel backend launch = false _mapreduce_nd_by_block!(
                src, dst, f, op, init, neutral, dims, Val(max_block_size)
            )

            workgroupsize = isnothing(block_size) ? KI.kernel_max_work_group_size(kernel) : block_size

            # One block per output element
            numworkgroups = dst_size
            kernel(
                src, dst, f, op, init, neutral, dims, Val(max_block_size); workgroupsize, numworkgroups
            )
        end
    end

    return dst
end


function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init,
    dims,
    max_tasks, min_elems,
)
    src_sizes = size(src)
    src_strides = strides(src)
    dst_strides = strides(dst)

    reduce_size = src_sizes[dims]
    ndims = length(src_sizes)

    # Each thread handles a section of the output array - i.e. reducing along the dims, for
    # multiple output strides
    foreachindex(dst, max_tasks=max_tasks, min_elems=min_elems) do idst

        @inbounds begin
            # Compute the base index in src (excluding the reduced axis)
            input_base_idx = 0
            tmp = idst - 1
            KernelAbstractions.Extras.@unroll for i in ndims:-1:1
                if i != dims
                    input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
                end
                tmp = tmp % dst_strides[i]
            end

            # Go over each element in the reduced dimension
            res = init
            for i in 0:reduce_size - 1
                src_idx = input_base_idx + i * src_strides[dims]
                res = op(res, f(src[src_idx + 1]))
            end
            dst[idst] = res
        end
    end

    dst
end


function _mapreduce_nd_by_thread!(
    src, dst,
    f, op,
    init, dims
)
    # One thread per output element, when there are more outer elements than in the reduced dim
    # e.g. reduce(+, rand(3, 1000), dims=1) => only 3 elements in the reduced dim
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = KI.get_group_id().x - 0x1
    ithread = KI.get_local_id().x - 0x1
    N = KI.get_local_size().x

    # Each thread handles one output element
    tid = ithread + iblock * N
    if tid < output_size

        # # Sometimes slightly faster method using additional memory with
        # # output_idx = @private typeof(iblock) (ndims,)
        # tmp = tid
        # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
        #     output_idx[i] = tmp ÷ dst_strides[i]
        #     tmp = tmp % dst_strides[i]
        # end
        # # Compute the base index in src (excluding the reduced axis)
        # input_base_idx = 0
        # KernelAbstractions.Extras.@unroll for i in 1:ndims
        #     i == dims && continue
        #     input_base_idx += output_idx[i] * src_strides[i]
        # end

        # Compute the base index in src (excluding the reduced axis)
        input_base_idx = typeof(ithread)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in ndims:-1i16:1i16
            if i != dims
                input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
            end
            tmp = tmp % dst_strides[i]
        end

        # Go over each element in the reduced dimension; this implementation assumes that there
        # are so many outer elements (each processed by an independent thread) that we afford to
        # loop sequentially over the reduced dimension (e.g. reduce(+, rand(3, 1000), dims=1))
        res = init
        for i in 0x0:reduce_size - 0x1
            src_idx = input_base_idx + i * src_strides[dims]
            res = op(res, f(src[src_idx + 0x1]))
        end
        dst[tid + 0x1] = res
    end
    nothing
end


function _mapreduce_nd_by_block!(
    src, dst,
    f, op,
    init, neutral,
    dims, ::Val{N}
) where N
    # One block per output element, when there are more elements in the reduced dim than in outer
    # e.g. reduce(+, rand(3, 1000), dims=2) => only 3 elements in outer dimensions
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    sdata = KI.localmemory(eltype(dst), N)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = KI.get_group_id().x - 0x1
    ithread = KI.get_local_id().x - 0x1

    # Each block handles one output element - thus, iblock ∈ [0, output_size)

    # # Sometimes slightly faster method using additional memory with
    # # output_idx = @private typeof(iblock) (ndims,)
    # tmp = iblock
    # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
    #     output_idx[i] = tmp ÷ dst_strides[i]
    #     tmp = tmp % dst_strides[i]
    # end
    # # Compute the base index in src (excluding the reduced axis)
    # input_base_idx = 0
    # KernelAbstractions.Extras.@unroll for i in 1:ndims
    #     i == dims && continue
    #     input_base_idx += output_idx[i] * src_strides[i]
    # end

    # Compute the base index in src (excluding the reduced axis)
    input_base_idx = typeof(ithread)(0)
    tmp = iblock
    KernelAbstractions.Extras.@unroll for i in ndims:-1i16:1i16
        if i != dims
            input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
        end
        tmp = tmp % dst_strides[i]
    end

    # We have a block of threads to process the whole reduced dimension. First do pre-reduction
    # in strides of N
    partial = neutral
    i = ithread
    while i < reduce_size
        src_idx = input_base_idx + i * src_strides[dims]
        partial = op(partial, f(src[src_idx + 0x1]))
        i += N
    end

    # Store partial result in shared memory; now we are down to a single block to reduce within
    sdata[ithread + 0x1] = partial
    KI.barrier()

    @inline reduce_group!(op, sdata, N, ithread)

    if ithread == 0x0
        dst[iblock + 0x1] = op(init, sdata[0x1])
    end
    nothing
end
