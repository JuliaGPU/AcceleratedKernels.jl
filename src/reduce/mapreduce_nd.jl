# Generalized N-dimensional mapreduce for GPU and CPU backends.

# 1. Dimension canonicalization: collapse adjacent reducible dims into contiguous segments
# 2. ND decode happens ONCE outside the inner loop using Val{sizes} (compile-time div → mulhi)
# 3. Inner loop is pure multiply-add, zero div/mod — compiler can vectorize
# 4. Multi-group reduction for large reduce spaces
# 5. Input staging (K elements per thread before shared memory)

# Host-side canonicalization

function _canonicalize_dims(src_sizes, src_strides, dims_valid)
    ndim = length(src_sizes)
    reduce_segs = Tuple{Int,Int}[]
    outer_segs  = Tuple{Int,Int}[]

    i = 1
    while i <= ndim
        if i in dims_valid
            seg_stride = src_strides[i]
            seg_size   = src_sizes[i]
            while i + 1 <= ndim &&
                  (i + 1) in dims_valid &&
                  src_strides[i + 1] == seg_stride * seg_size
                i += 1
                seg_size *= src_sizes[i]
            end
            push!(reduce_segs, (seg_stride, seg_size))
        else
            push!(outer_segs, (src_strides[i], src_sizes[i]))
        end
        i += 1
    end

    return Tuple(reduce_segs), Tuple(outer_segs)
end

# Main entry point

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

    dims_all = dims isa Int ? (dims,) : dims

    if Base.any(d < 1 for d in dims_all)
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    src_sizes   = size(src)
    src_strides = strides(src)
    ndim        = length(src_sizes)

    dims_valid = Tuple(d for d in dims_all if d <= ndim)

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

    dst_sizes = unrolled_map_index(src_sizes) do i
        i in dims_valid ? 1 : src_sizes[i]
    end

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

    len = Base.prod(src_sizes[d] for d in dims_valid)

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
        _mapreduce_nd_cpu_sections!(f, op, dst, src; init, dims=dims_valid, max_tasks, min_elems)
    else
        reduce_segs, outer_segs = _canonicalize_dims(src_sizes, src_strides, dims_valid)

        outer_sizes_tup    = Tuple(s   for (_, s)   in outer_segs)
        outer_strides_tup  = Tuple(str for (str, _) in outer_segs)
        reduce_sizes_tup   = Tuple(s   for (_, s)   in reduce_segs)
        reduce_strides_tup = Tuple(str for (str, _) in reduce_segs)

        reduce_size   = len
        # Multi-group heuristic:
        # - Always use if reduce_size is very large (> 16 * block_size)
        # - Skip if dst_size is tiny AND reduce_size is moderate (overhead dominates)
        raw_groups = (reduce_size + block_size - 1) ÷ block_size
        reduce_groups = if raw_groups <= 1
            1
        elseif dst_size >= block_size
            raw_groups  # large output: always multi-group
        elseif reduce_size > 16 * block_size
            raw_groups  # large reduction: need multi-group regardless
        else
            1           # small output + moderate reduction: single-group wins
        end

        if reduce_groups == 1
            if dst_size >= reduce_size
                blocks = (dst_size + block_size - 1) ÷ block_size
                kernel! = _mapreduce_nd_by_thread!(backend, block_size)
                kernel!(
                    src, dst, f, op, init,
                    outer_strides_tup, outer_sizes_tup,
                    reduce_strides_tup, reduce_sizes_tup,
                    dst_size, reduce_size,
                    ndrange=(block_size * blocks,),
                )
            else
                kernel! = _mapreduce_nd_by_block!(backend, block_size)
                kernel!(
                    src, dst, f, op, init, neutral,
                    outer_strides_tup, outer_sizes_tup,
                    reduce_strides_tup, reduce_sizes_tup,
                    dst_size, reduce_size,
                    ndrange=(block_size * dst_size,),
                )
            end
        else
            partial = KernelAbstractions.allocate(backend, typeof(init), (dst_size, reduce_groups))

            kernel! = _mapreduce_nd_multigroup!(backend, block_size)
            kernel!(
                src, partial, f, op, neutral,
                outer_strides_tup, outer_sizes_tup,
                reduce_strides_tup, reduce_sizes_tup,
                dst_size, reduce_size, reduce_groups,
                ndrange=(block_size * dst_size * reduce_groups,),
            )

            # Second pass: reduce partial (dst_size × reduce_groups) → dst
            if reduce_groups <= block_size
                # Small enough: one block handles it sequentially — low overhead
                kernel2! = _mapreduce_partial_to_dst!(backend, block_size)
                kernel2!(
                    partial, dst, op, init,
                    dst_size, reduce_groups,
                    ndrange=(block_size * dst_size,),
                )
            else
                # Large: recurse with proper block reduction
                dst_2d = reshape(dst, (dst_size, 1))
                mapreduce_nd(
                    identity, op, partial, backend;
                    init,
                    neutral,
                    dims=2,
                    max_tasks, min_elems, prefer_threads,
                    block_size,
                    temp=dst_2d,
                )
            end
        end
    end

    return dst
end

# CPU path

function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init, dims, max_tasks, min_elems,
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

# Index helpers

@inline function _outer_decode(tid, outer_strides, outer_sizes)
    isempty(outer_sizes) && return 0
    base = 0
    tmp  = tid
    @inbounds for i in 1:length(outer_sizes)
        q    = tmp ÷ outer_sizes[i]
        r    = tmp - q * outer_sizes[i]
        base += r * outer_strides[i]
        tmp   = q
    end
    base
end

@inline function _reduce_offset(j, reduce_strides, reduce_sizes)
    isempty(reduce_sizes) && return 0
    off = 0
    tmp = j
    @inbounds for i in 1:length(reduce_sizes)
        q   = tmp ÷ reduce_sizes[i]
        r   = tmp - q * reduce_sizes[i]
        off += r * reduce_strides[i]
        tmp  = q
    end
    off
end

# GPU kernel: by_thread: one thread per output element

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread!(
    @Const(src), dst,
    f, op, init,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size,
)
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < output_size
        input_base = _outer_decode(tid, outer_strides, outer_sizes)

        res = init
        for j in 0x0:reduce_size - 0x1
            off = _reduce_offset(j, reduce_strides, reduce_sizes)
            res = op(res, f(src[input_base + off + 0x1]))
        end

        dst[tid + 0x1] = res
    end
end

# GPU kernel: by_block: one block per output element, single group

const _STAGING = 4

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_block!(
    @Const(src), dst,
    f, op, init, neutral,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        input_base = _outer_decode(iblock, outer_strides, outer_sizes)

        acc = neutral
        j   = ithread
        while j < reduce_size
            KernelAbstractions.Extras.@unroll for k in 1:_STAGING
                if j < reduce_size
                    off = _reduce_offset(j, reduce_strides, reduce_sizes)
                    acc = op(acc, f(src[input_base + off + 0x1]))
                    j  += N
                end
            end
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end

# GPU kernel: multi-group

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_multigroup!(
    @Const(src), partial,
    f, op, neutral,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size, reduce_groups,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(partial) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    iout   = iblock % output_size
    igroup = iblock ÷ output_size

    input_base = _outer_decode(iout, outer_strides, outer_sizes)

    acc = neutral
    j   = ithread + igroup * N
    while j < reduce_size
        KernelAbstractions.Extras.@unroll for k in 1:_STAGING
            if j < reduce_size
                off = _reduce_offset(j, reduce_strides, reduce_sizes)
                acc = op(acc, f(src[input_base + off + 0x1]))
                j  += N * reduce_groups
            end
        end
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    if ithread == 0x0
        partial[iout + igroup * output_size + 0x1] = sdata[0x1]
    end
end

# GPU kernel: second pass — partial → dst

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_partial_to_dst!(
    @Const(partial), dst,
    op, init,
    output_size, reduce_groups,
)
    # One block per output element — block reduces over reduce_groups
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        # Each thread strides over reduce_groups
        acc = eltype(dst)(0)  # neutral for this pass — init applied at end
        g = ithread
        while g < reduce_groups
            acc = op(acc, partial[iblock + g * output_size + 0x1])
            g += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end
