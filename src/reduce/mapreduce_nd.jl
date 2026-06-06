# Generalized N-dimensional mapreduce for GPU and CPU backends.
# Uses stride-arithmetic-based indexing (no CartesianIndices in GPU kernels)
# with multi-group reduction for large reduction dimensions.
# Note: This file was developed with AI assistance (Claude, Anthropic).

function mapreduce_nd(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(src)),
    dims::Union{Int, Tuple{Vararg{Int}}},

    # CPU settings
    max_tasks::Int,
    min_elems::Int,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int,
    temp::Union{Nothing, AbstractArray},
)
    @argcheck 1 <= block_size <= 1024

    # Normalize dims to a tuple
    dims_all = dims isa Int ? (dims,) : dims

    # Invalid dims: negative or zero
    if Base.any(d < 1 for d in dims_all)
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    src_sizes = size(src)
    ndim = length(src_sizes)

    # Filter out-of-range dims (Base silently ignores them)
    # e.g. dims=(1,4) on a 3D array → only dim 1 is reduced
    dims_valid = Tuple(d for d in dims_all if d <= ndim)

    # If ALL dims are out of range, just map each element through f and add init
    if isempty(dims_valid)
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

    # Destination sizes: reduced dims become 1
    dst_sizes = unrolled_map_index(src_sizes) do i
        i in dims_valid ? 1 : src_sizes[i]
    end

    # If any kept dimension is zero → return empty array
    for isize in eachindex(src_sizes)
        isize in dims_valid && continue
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

    # Total number of elements being reduced per output element
    len = Base.prod(src_sizes[d] for d in dims_valid)

    # If reduced dims are all zero → fill with init
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

    # If reduced dims are all 1 → just apply f element-wise
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

    # Allocate destination
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
            dims=dims_valid,
            max_tasks=max_tasks,
            min_elems=min_elems,
        )
    else
        # Precompute strides on host — passed as tuples to kernel
        # This avoids CartesianIndices div/mod inside the kernel
        src_str = strides(src)
        dst_str = strides(dst)

        # reduce_strides: for each src dim, its stride if it's a reduced dim, else 0
        # Used to walk through the reduced subspace
        reduce_str = unrolled_map_index(src_str) do i
            i in dims_valid ? src_str[i] : 0
        end

        if dst_size >= len
            # More output elements than reduction elements → one thread per output
            blocks = (dst_size + block_size - 1) ÷ block_size
            kernel1! = _mapreduce_nd_by_thread!(backend, block_size)
            kernel1!(
                src, dst, f, op, init,
                src_str, dst_str, reduce_str,
                dst_size, len, ndim,
                ndrange=(block_size * blocks,),
            )
        else
            # More reduction elements than output elements → one block per output
            reduce_groups = (len + block_size - 1) ÷ block_size

            if reduce_groups == 1
                # Single group — all threads in one block handle one output element
                kernel2! = _mapreduce_nd_by_block!(backend, block_size)
                kernel2!(
                    src, dst, f, op, init, neutral,
                    src_str, dst_str, reduce_str,
                    dst_size, len, ndim,
                    ndrange=(block_size * dst_size,),
                )
            else
                # Multi-group — multiple blocks per output element
                # partial shape: (dst_sizes..., reduce_groups)
                partial_sizes = (dst_sizes..., reduce_groups)
                partial = KernelAbstractions.allocate(backend, typeof(init), partial_sizes)

                partial_str = strides(partial)

                kernel3! = _mapreduce_nd_by_block_multigroup!(backend, block_size)
                kernel3!(
                    src, partial, f, op, neutral,
                    src_str, dst_str, reduce_str, partial_str,
                    dst_size, len, reduce_groups, ndim,
                    ndrange=(block_size * dst_size * reduce_groups,),
                )

                # Second pass: reduce partial → dst
                # partial has shape (dst_sizes..., reduce_groups)
                # treat last dim as the reduction dim
                partial_dst_str = strides(partial)[1:ndim]  # strides of first ndim dims
                reduce_str2 = unrolled_map_index(strides(partial)) do i
                    i == ndim + 1 ? strides(partial)[ndim + 1] : 0
                end

                kernel4! = _mapreduce_nd_by_block!(backend, block_size)
                kernel4!(
                    partial, dst, identity, op, init, neutral,
                    strides(partial), dst_str, reduce_str2,
                    dst_size, reduce_groups, ndim + 1,
                    ndrange=(block_size * dst_size,),
                )
            end
        end
    end

    return dst
end


# CPU path — CartesianIndices is fine here
function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init,
    dims,
    max_tasks, min_elems,
)
    Rother  = CartesianIndices(dst)
    Rreduce = CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))

    foreachindex(dst, max_tasks=max_tasks, min_elems=min_elems) do idst
        @inbounds begin
            Iother = Rother[idst]
            res = init
            for Ireduce in Rreduce
                J = max(Iother, Ireduce)
                res = op(res, f(src[J]))
            end
            dst[idst] = res
        end
    end

    dst
end


# GPU kernel: one thread per output element
# Uses stride arithmetic — no CartesianIndices
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread!(
    @Const(src), dst,
    f, op, init,
    src_str, dst_str, reduce_str,
    output_size, reduce_size, ndim,
)
    N = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid = ithread + iblock * N

    if tid < output_size
        # Compute base index in src for this output element
        # Walk dst linear index → per-dim indices → src offset (skip reduced dims)
        input_base_idx = typeof(ithread)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
            if dst_str[i] > 0 && reduce_str[i] == 0
                # This is a kept dim
                dim_idx = tmp ÷ dst_str[i]
                input_base_idx += dim_idx * src_str[i]
            end
            tmp = tmp % dst_str[i]
        end

        # Walk through reduced subspace using reduce_str
        res = init
        for j in 0x0:reduce_size - 0x1
            reduce_offset = typeof(ithread)(0)
            tmp2 = j
            KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
                if reduce_str[i] > 0
                    reduce_offset += (tmp2 ÷ (reduce_str[i] ÷ src_str[1])) * src_str[i]
                    tmp2 = tmp2 % (reduce_str[i] ÷ src_str[1])
                end
            end
            res = op(res, f(src[input_base_idx + reduce_offset + 0x1]))
        end

        dst[tid + 0x1] = res
    end
end


# GPU kernel: one block per output element, single group
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_block!(
    @Const(src), dst,
    f, op, init, neutral,
    src_str, dst_str, reduce_str,
    output_size, reduce_size, ndim,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        # Compute base index in src for this output element
        input_base_idx = typeof(ithread)(0)
        tmp = iblock
        KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
            if reduce_str[i] == 0
                dim_idx = tmp ÷ dst_str[i]
                input_base_idx += dim_idx * src_str[i]
            end
            tmp = tmp % dst_str[i]
        end

        # Each thread reduces a strided slice of the reduce space
        partial = neutral
        j = ithread
        while j < reduce_size
            reduce_offset = typeof(ithread)(0)
            tmp2 = j
            KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
                if reduce_str[i] > 0
                    s = reduce_str[i] ÷ src_str[1]
                    reduce_offset += (tmp2 ÷ s) * src_str[i]
                    tmp2 = tmp2 % s
                end
            end
            partial = op(partial, f(src[input_base_idx + reduce_offset + 0x1]))
            j += N
        end

        sdata[ithread + 0x1] = partial
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end


# GPU kernel: multi-group reduction — multiple blocks per output element
# Writes partial results to a (dst_sizes..., reduce_groups) array
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_block_multigroup!(
    @Const(src), partial,
    f, op, neutral,
    src_str, dst_str, reduce_str, partial_str,
    output_size, reduce_size, reduce_groups, ndim,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(partial) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # iblock encodes both which output element and which reduce group
    iout   = iblock % output_size          # which output element
    igroup = iblock ÷ output_size          # which reduce group

    # Compute base index in src for this output element
    input_base_idx = typeof(ithread)(0)
    tmp = iout
    KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
        if reduce_str[i] == 0
            dim_idx = tmp ÷ dst_str[i]
            input_base_idx += dim_idx * src_str[i]
        end
        tmp = tmp % dst_str[i]
    end

    # Each group handles a chunk of the reduce space
    chunk_start = igroup * N + ithread    # starting position in reduce space
    chunk_stride = N * reduce_groups      # stride across groups

    acc = neutral
    j = chunk_start
    while j < reduce_size
        reduce_offset = typeof(ithread)(0)
        tmp2 = j
        KernelAbstractions.Extras.@unroll for i in ndim:-1i16:1i16
            if reduce_str[i] > 0
                s = reduce_str[i] ÷ src_str[1]
                reduce_offset += (tmp2 ÷ s) * src_str[i]
                tmp2 = tmp2 % s
            end
        end
        acc = op(acc, f(src[input_base_idx + reduce_offset + 0x1]))
        j += chunk_stride
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    if ithread == 0x0
        # Write to partial[(iout linear), igroup+1]
        partial_idx = iout + igroup * output_size
        partial[partial_idx + 0x1] = sdata[0x1]
    end
end
