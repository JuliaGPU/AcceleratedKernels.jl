# Generalized N-dimensional mapreduce for GPU and CPU backends, reducing one or more
# dimensions (`dims::Int` or `dims::Tuple`) of `src` into `dst`.
#
# Design (see references: CUDA.jl / GPUArrays mapreducedim!, PyTorch Reduce.cuh, CUB):
#   1. Canonicalize dims: collapse adjacent dimensions with matching strides into contiguous
#      segments. A plain `dims=2` reduction, or any reduction over a contiguous block of dims
#      (e.g. `dims=(1,2)`), collapses to a *single* reduce segment.
#   2. The inner reduce loop must avoid per-element integer division. For a single segment the
#      element offset is just `j * stride`; only genuinely non-contiguous dim sets (e.g.
#      `dims=(1,3)`) fall back to a per-element multi-dimensional decode.
#   3. Three work decompositions, chosen by the relative sizes of the output and the reduction:
#        - by_thread:  one thread per output (many outputs, small reduction)
#        - by_block:   one block per output  (few outputs, large reduction)
#        - multigroup: several blocks per output, two-pass (very few outputs, huge reduction)

# Number of first-pass blocks the multi-group reduction aims to launch, so a reduction with
# very few output elements can still fill the GPU. A heuristic GPU-occupancy target; the
# multi-group path is never used unless it launches at least as many blocks as by_block would.
const TARGET_BLOCKS = 256


# Host-side canonicalization: split the dimensions into reduced and kept ("outer") segments,
# merging adjacent dimensions whose strides are contiguous. Returns two tuples of
# (stride, size) segments.
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

    # Duplicate dims check: Base errors on dims=(2,2)
    if length(dims_all) != length(Base.unique(dims_all))
        throw(ArgumentError("region dimension(s) must be unique, got $dims"))
    end

    src_sizes   = size(src)
    src_strides = strides(src)
    ndim        = length(src_sizes)

    dims_valid = Tuple(d for d in dims_all if d <= ndim)

    # Degenerate cases begin; order of priority matters

    # All reduced dims are beyond ndims: just map each element through f and add init, e.g.:
    #   julia> x = rand(Float64, 3, 5);
    #   julia> mapreduce(x -> -x, +, x, dims=3, init=Float32(0))     # 3×5 Matrix{Float32}
    if isempty(dims_valid)
        dst = _alloc_or_temp(backend, temp, init, src_sizes)
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i in dims_valid ? 1 : src_sizes[i]
    end

    # If any kept dimension is zero, return empty array (reduced dims become 1), e.g.:
    #   julia> x = rand(3, 0, 5);  reduce(+, x, dims=3)     # 3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize in dims_valid && continue
        if src_sizes[isize] == 0
            return _alloc_or_temp(backend, temp, init, dst_sizes)
        end
    end

    len = Base.prod(src_sizes[d] for d in dims_valid)

    # If a reduced dimension is zero, return array filled with init, e.g.:
    #   julia> x = rand(3, 0, 5);  mapreduce(+, x, dims=2)     # 3×1×5 of zeros
    if len == 0
        dst = _alloc_or_temp(backend, temp, init, dst_sizes)
        fill!(dst, init)
        return dst
    end

    # If the reduced extent is 1, just map each element through f (keep init's type)
    if len == 1
        dst = _alloc_or_temp(backend, temp, init, src_sizes)
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # Degenerate cases end

    dst = _alloc_or_temp(backend, temp, init, dst_sizes)
    dst_size = length(dst)

    if !use_gpu_algorithm(backend, prefer_threads)
        _mapreduce_nd_cpu_sections!(f, op, dst, src; init, max_tasks, min_elems)
        return dst
    end

    reduce_segs, outer_segs = _canonicalize_dims(src_sizes, src_strides, dims_valid)

    outer_strides  = Tuple(str for (str, _) in outer_segs)
    outer_sizes    = Tuple(s   for (_, s)   in outer_segs)
    reduce_strides = Tuple(str for (str, _) in reduce_segs)
    reduce_sizes   = Tuple(s   for (_, s)   in reduce_segs)
    reduce_size    = len

    # One block per output (by_block) launches `dst_size` blocks. When there are too few
    # outputs to fill the GPU *and* the reduction is large, split each output's reduction
    # across `reduce_groups` blocks and combine the partials in a cheap second pass. Capping
    # at `block_size` keeps that second pass to a single block per output.
    reduce_groups = 1
    if dst_size < reduce_size && dst_size < TARGET_BLOCKS
        reduce_groups = min(cld(reduce_size, block_size), block_size, cld(TARGET_BLOCKS, dst_size))
    end

    if reduce_groups > 1
        partial = KernelAbstractions.allocate(backend, typeof(init), (dst_size, reduce_groups))

        kernel! = _mapreduce_nd_multigroup!(backend, block_size)
        kernel!(
            src, partial, f, op, neutral,
            outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, reduce_groups,
            ndrange=(block_size * dst_size * reduce_groups,),
        )

        # Second pass: reduce partial (dst_size × reduce_groups) → dst, one block per output
        kernel2! = _mapreduce_partial_to_dst!(backend, block_size)
        kernel2!(
            partial, dst, op, init, neutral,
            dst_size, reduce_groups,
            ndrange=(block_size * dst_size,),
        )
    elseif dst_size >= reduce_size
        # Many outputs, small reduction: one thread per output reduces sequentially
        blocks = cld(dst_size, block_size)
        kernel! = _mapreduce_nd_by_thread!(backend, block_size)
        kernel!(
            src, dst, f, op, init,
            outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    else
        # Few outputs, large reduction: one block of threads cooperatively reduces each output
        kernel! = _mapreduce_nd_by_block!(backend, block_size)
        kernel!(
            src, dst, f, op, init, neutral,
            outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * dst_size,),
        )
    end

    return dst
end


# Allocate a destination array, or validate and reuse a user-provided `temp`.
function _alloc_or_temp(backend, temp, init, sizes)
    isnothing(temp) && return KernelAbstractions.allocate(backend, typeof(init), sizes)
    @argcheck get_backend(temp) == backend
    @argcheck size(temp) == sizes
    @argcheck eltype(temp) == typeof(init)
    temp
end

# CPU path

function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init, max_tasks, min_elems,
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

# Index helpers. Both decode a linear index into a byte-offset by walking compile-time-sized
# segment tuples. The single-segment case (the common one after canonicalization) needs no
# division; multi-segment falls back to a per-element decode (rare: non-adjacent dim sets).

@inline function _outer_decode(tid, outer_strides, outer_sizes)
    isempty(outer_sizes) && return 0
    length(outer_sizes) == 1 && return tid * outer_strides[1]
    base = 0
    tmp  = tid
    @inbounds for i in 1:length(outer_sizes)
        q     = tmp ÷ outer_sizes[i]
        r     = tmp - q * outer_sizes[i]
        base += r * outer_strides[i]
        tmp   = q
    end
    base
end

@inline function _reduce_offset(j, reduce_strides, reduce_sizes)
    isempty(reduce_sizes) && return 0
    # Single segment: j < reduce_size, so the decode is just j * stride — no division.
    length(reduce_sizes) == 1 && return j * reduce_strides[1]
    off = 0
    tmp = j
    @inbounds for i in 1:length(reduce_sizes)
        q    = tmp ÷ reduce_sizes[i]
        r    = tmp - q * reduce_sizes[i]
        off += r * reduce_strides[i]
        tmp  = q
    end
    off
end

# GPU kernel: by_thread — one thread per output element, reducing sequentially.
# Used when there are more output elements than elements in the reduced dimension(s),
# e.g. reduce(+, rand(3, 1000), dims=1) — only 3 elements to reduce per output.

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread!(
    @Const(src), dst,
    f, op, init,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size,
)
    # NOTE: index calculations use zero-indexing (fewer ops, matches the CUDA / ROCm / oneAPI /
    # Metal code this is transpiled to), converting to one-indexing only at memory accesses.
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

# GPU kernel: by_block — one block of threads cooperatively reduces each output element.
# Used when there are more elements in the reduced dimension(s) than output elements,
# e.g. reduce(+, rand(3, 1000), dims=2) — only 3 output elements.

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

        # Pre-reduce in strides of N (consecutive threads read consecutive elements)
        acc = neutral
        j   = ithread
        while j < reduce_size
            off = _reduce_offset(j, reduce_strides, reduce_sizes)
            acc = op(acc, f(src[input_base + off + 0x1]))
            j  += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end

# GPU kernel: multi-group first pass — several blocks per output element. Block `iblock`
# handles output `iout` and group `igroup`, reducing its interleaved slice of the reduction
# into partial[iout, igroup]. The second pass combines the groups.

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
        off = _reduce_offset(j, reduce_strides, reduce_sizes)
        acc = op(acc, f(src[input_base + off + 0x1]))
        j  += N * reduce_groups
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    if ithread == 0x0
        partial[iout + igroup * output_size + 0x1] = sdata[0x1]
    end
end

# GPU kernel: multi-group second pass — one block per output reduces over the `reduce_groups`
# partials and folds in `init`.

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_partial_to_dst!(
    @Const(partial), dst,
    op, init, neutral,
    output_size, reduce_groups,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        acc = neutral
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
