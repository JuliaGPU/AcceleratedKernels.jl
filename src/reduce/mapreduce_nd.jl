# Generalized N-dimensional mapreduce for GPU and CPU backends, reducing one or more
# dimensions (`dims::Integer` or a collection of integers) of `src` into `dst`.
#
# Design (see references: CUDA.jl / GPUArrays mapreducedim!, PyTorch Reduce.cuh, CUB):
#   1. Canonicalize dims: collapse adjacent dimensions with matching strides into contiguous
#      segments. A plain `dims=2` reduction, or any reduction over a contiguous block of dims
#      (e.g. `dims=(1,2)`), collapses to a *single* reduce segment.
#   2. The inner reduce loop must avoid per-element integer division. For a single segment the
#      element offset is just `j * stride`; only genuinely non-contiguous dim sets (e.g.
#      `dims=(1,3)`) fall back to a per-element multi-dimensional decode.
#   3. Six work decompositions, chosen by the relative sizes and strides of the output and
#      the reduction:
#        - by_thread:     one thread per output (many outputs, small reduction)
#        - columns:       32 contiguous outputs per block, split across blocks, for long strided
#                         reductions (e.g. `dims=2` of a wide matrix)
#        - tiled_strided: several contiguous outputs per block, for square-like strided reductions
#        - by_block:      one block per output, grid-stride over outputs (few outputs, large reduction)
#        - multigroup:    several blocks per output, two-pass (dst_size==1 or very small dst_size)
#        - segments:      a group of lanes per output, several outputs per block, for reductions
#                         of consecutive elements

# Main entry point

# The reduced dimensions `dims` (an integer or a collection of integers), as Base validates them:
# duplicates are ignored, and dimensions beyond `ndims` reduce nothing.
function _reduced_dims(dims, ndim::Int)
    dims_src = dims isa Number ? (dims,) : dims
    dims_buf = Int[]
    for d in dims_src
        d isa Integer || throw(ArgumentError("reduced dimension(s) must be integers"))
        dim = Int(d)
        dim < 1 && throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
        push!(dims_buf, dim)
    end
    return Tuple(d for d in Base.unique(dims_buf) if d <= ndim)
end

# Reduce `src` into `dst`, which has `src`'s number of dimensions and size 1 along the reduced
# dimensions `dims_valid`; `A` is the accumulator type, and `init` is a value, `_NoInit()` or
# `_Fold()` (see `_finish`). Without `commutative`, the GPU kernels combine elements in order.
function mapreduce_nd!(
    f::F, op::OP, dst::AbstractArray, src::MapReduceSource, backend::Backend,
    alg::Union{BlockReduce, CPUThreads.Partitioned}, ::Type{A};
    init,
    neutral,
    dims_valid::Tuple,
    bufs::NamedTuple,
    commutative::Bool=true,
) where {F, OP, A}
    # Launch settings of the elementwise passes, and of the reduction kernels
    launch = alg isa BlockReduce ? (; block_size=alg.block_size) :
                                   (; max_tasks=alg.max_tasks, min_elems=alg.min_elems)

    # Number of blocks the by_block / multigroup paths aim to launch, so a reduction with
    # few output elements can still fill the GPU. A heuristic GPU-occupancy target.
    tuning = reduce_tuning(backend, A)
    target_blocks = tuning.target_blocks
    target_blocks >= 1 || throw(ArgumentError(
        "the reduction tuning's `target_blocks` must be positive, got $target_blocks"))

    src_sizes = size(src)

    # Degenerate cases begin; order of priority matters

    len = Base.prod(src_sizes[d] for d in dims_valid; init=1)

    # Empty reductions, e.g. reduce(+, rand(3, 0, 5), dims=2): `init`, else nothing is written
    if len == 0
        init isa Union{_NoInit, _Fold} || _fill!(dst, init, backend)
        return dst
    end

    # Nothing to write, e.g. reduce(+, rand(3, 0, 5), dims=3), a 3×0×1 array
    isempty(dst) && return dst
    # (overwriting with single elements combines nothing, even when `op` always throws)
    len == 1 && init isa _NoInit || _check_acctype(op, f, A)

    # Every output reduces one element, e.g. reduce(+, rand(3, 5), dims=3): no reduced
    # dimensions, or only ones of size 1
    if len == 1
        # (a runtime dispatch on the GPU, see `_mapreduce_nd_launch!`)
        single! = alg isa BlockReduce ? Base.inferencebarrier(_mapreduce_nd_single!) :
                                        _mapreduce_nd_single!
        single!(f, op, dst, src, backend, A; init, launch...)
        return dst
    end

    # Degenerate cases end

    dst_size = length(dst)

    if alg isa CPUThreads.Partitioned
        _mapreduce_nd_cpu_sections!(f, op, dst, src; init, neutral, launch...)
        return dst
    end
    block_size = alg.block_size

    # A single output of a contiguous source: the whole-array kernels, finishing on the device
    if dst_size == 1
        linear = _mapreduce_nd_linear(src, dims_valid)
        if linear !== nothing
            Base.inferencebarrier(mapreduce_1d_gpu)(
                f, op, linear, backend; init, neutral, block_size,
                items_per_thread=alg.items_per_thread, max_blocks=target_blocks,
                partials=get(bufs, :partials, nothing), switch_below=0, commutative, dst)
            return dst
        end
    end

    # The kernels below index a flat buffer at `buffer[base_offset + Σ coordᵈ·strideᵈ + 1]`
    # (see `_mapreduce_layout`)
    layout = _mapreduce_strided_layout(src)
    buffer, base_offset, src_strides =
        isnothing(layout) ? _mapreduce_linear_layout(src) : layout
    segs = _mapreduce_segments(src_sizes, src_strides, dims_valid)
    reduce_size = len
    shape, reduce_groups = _mapreduce_nd_shape(dst_size, reduce_size, segs, block_size, tuning, A)
    commutative || ((shape, reduce_groups) = _ordered_shape(shape, reduce_groups, dst_size,
                                                            reduce_size, block_size))

    # One output per thread of a source without a dense buffer: iterating over the Cartesian
    # indices of each output's elements saves decoding every linear index
    if isnothing(layout) && shape === :by_thread
        blocks = cld(dst_size, block_size)
        kernel! = _mapreduce_nd_generic!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            src, dst, f, op, neutral, init,
            CartesianIndices(dst), _mapreduce_reduce_indices(src, dst), dst_size,
            ndrange=(block_size * blocks,),
        )
        return dst
    end

    _with_segments(segs) do reduce_strides, reduce_sizes, outer_strides, outer_sizes
        Base.inferencebarrier(_mapreduce_nd_launch!)(
            Val(shape), reduce_groups, f, op, dst, buffer, backend, block_size, target_blocks;
            init, neutral, bufs, base_offset, reduce_strides, reduce_sizes, outer_strides,
            outer_sizes, dst_size, reduce_size, dense_buffer=!isnothing(layout),
            items_per_thread=alg.items_per_thread,
        )
    end

    return dst
end

# Reductions whose every output reduces a single element, a partial result of the accumulator
# type `A`: `dst` and `src` have the same length.
function _mapreduce_nd_single!(f::F, op::OP, dst, src, backend, ::Type{A}; init,
                               launch...) where {F, OP, A}
    _foreachindex(eachindex(dst), backend; launch...) do i
        x = Base.mapreduce_first(f, op, src[i])
        dst[i] = _finish(op, init, dst, i, A === Union{} ? x : convert(A, x))
    end
end

# Launch the kernels of a strided reduction of shape `shape` (see `_mapreduce_nd_shape`).
#
# This and the other GPU launches of a reduction are called through `Base.inferencebarrier`.
# Inferring a launch through KernelAbstractions and the back-end takes far longer than the launch,
# and without the barrier a reduction with a new `f`, `op` or element type infers every launch it
# could make: all shapes, segment layouts and vector widths. A dynamic dispatch costs a fraction of
# a microsecond.
#
# `dense_buffer` is whether `buffer` is the dense buffer of the source, which the by-block kernel
# can read with vector loads, or a linearly indexed lazy source (see `_mapreduce_layout`).
function _mapreduce_nd_launch!(
    ::Val{shape}, reduce_groups, f::F, op::OP, dst, buffer, backend, block_size, target_blocks;
    init, neutral, bufs, base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes,
    dst_size, reduce_size, dense_buffer, items_per_thread,
) where {shape, F, OP}
    if shape === :ordered
        partials = reduce_groups > 1 ? bufs.partials : nothing
        _launch_mapreduce_ordered!(
            backend, block_size, items_per_thread, buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, reduce_groups, partials; dense_buffer,
        )
    elseif shape === :columns || shape === :columns_ordered
        ordered = shape === :columns_ordered
        groups = reduce_groups
        # (runs of `outer_sizes[1]` contiguous outputs, at the other kept dimensions' offsets)
        run = outer_sizes[1]
        runs = Base.tail(outer_strides), Base.tail(outer_sizes)
        cols = _columns_cols(run)
        blocks = cld(run, cols) * (dst_size ÷ run) * groups
        threads = min(block_size, COLUMNS_MAX_THREADS)
        kernel! = _mapreduce_nd_columns!(backend, threads)
        if groups == 1
            kernel!(buffer, dst, f, op, neutral, init, base_offset, reduce_strides[1], dst_size,
                    run, dst_size ÷ run, runs..., reduce_size, 1, Val(cols), Val(ordered);
                    ndrange=(threads * blocks,))
        else
            # (the partial results of each output's groups, combined by a second pass)
            partial = bufs.partials
            kernel!(buffer, partial, f, op, neutral, _NoFinish(), base_offset,
                    reduce_strides[1], dst_size, run, dst_size ÷ run, runs..., reduce_size, groups,
                    Val(cols), Val(ordered);
                    ndrange=(threads * blocks,))
            pass2_block_size = _pass2_block_size(groups)
            if ordered
                _mapreduce_ordered_pass!(backend, pass2_block_size, 1, partial, dst, _Partials(),
                                         op, neutral, init, 0, (1,), (dst_size,), (dst_size,),
                                         (groups,), dst_size, groups, 1)
            else
                kernel2! = _mapreduce_partial_to_dst!(backend, pass2_block_size)
                kernel2!(partial, dst, op, neutral, init, dst_size, groups,
                         ndrange=(pass2_block_size * dst_size,))
            end
        end
    elseif shape === :segments
        lanes = reduce_groups
        blocks = cld(dst_size, block_size ÷ lanes)
        kernel! = _mapreduce_nd_segments!(backend, block_size)
        kernel!(
            buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes, dst_size, reduce_size, Val(lanes),
            ndrange=(block_size * blocks,),
        )
    elseif shape === :tiled_strided
        # Narrow layout-specific path: square-like row reductions with contiguous
        # outputs and strided input reads, e.g. size=(1024,1024), dims=2. Several
        # outputs share a block: small lane groups reduce one output each, preserving
        # more cross-output memory coalescing than by_block while launching many more
        # blocks than by_thread.
        rows_per_block = TILED_STRIDED_ROWS_PER_BLOCK
        blocks = cld(dst_size, rows_per_block)
        kernel! = _mapreduce_nd_by_thread_tiled_strided!(backend, block_size)
        kernel!(
            buffer, dst, f, op, neutral, init,
            base_offset, reduce_strides[1], dst_size, reduce_size, Val(rows_per_block),
            ndrange=(block_size * blocks,),
        )
    elseif shape === :by_thread
        # Many outputs, small reduction: one thread per output reduces sequentially
        blocks = cld(dst_size, block_size)
        kernel! = _mapreduce_nd_by_thread!(backend, block_size)
        kernel!(
            buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    elseif shape === :multigroup
        # Very few outputs, large reduction: split each output's reduction across
        # `reduce_groups` blocks, combine the partials in a second pass
        partial = bufs.partials
        kernel! = _mapreduce_nd_multigroup!(backend, block_size)
        kernel!(
            buffer, partial, f, op, neutral,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            Val(dst_size), reduce_size, reduce_groups,
            ndrange=(block_size * dst_size * reduce_groups,),
        )

        # Second pass: reduce partial (dst_size × reduce_groups) → dst, one block
        # per output. reduce_groups is small (<=block_size by construction), so use
        # a small block size for this pass to avoid an oversubscribed tree-reduce.
        pass2_block_size = _pass2_block_size(reduce_groups)
        kernel2! = _mapreduce_partial_to_dst!(backend, pass2_block_size)
        kernel2!(
            partial, dst, op, neutral, init,
            dst_size, reduce_groups,
            ndrange=(pass2_block_size * dst_size,),
        )
    else
        # One block per output: grid-striding over the outputs with at most
        # `target_blocks` blocks, or (a multigroup reduction that collapsed to one group,
        # e.g. reduce_size <= block_size) one block for each of the few outputs
        launch_blocks = shape === :by_block ? min(dst_size, target_blocks) : dst_size
        _launch_mapreduce_nd_by_block!(
            backend, block_size, buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, launch_blocks, dense_buffer,
        )
    end

    return nothing
end

function _launch_mapreduce_nd_by_block!(
    backend, block_size, buffer, dst, f::F, op::OP, neutral, init,
    base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
    output_size, reduce_size, num_blocks, dense_buffer,
) where {F, OP}
    W = dense_buffer ? _contiguous_vector_width(
        buffer, base_offset, outer_strides, reduce_strides, reduce_size, block_size,
    ) : 0
    if W == 4
        kernel! = _mapreduce_nd_by_block_contiguous!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes,
            output_size, reduce_size, num_blocks, Val(4),
            ndrange=(block_size * num_blocks,),
        )
    elseif W == 2
        kernel! = _mapreduce_nd_by_block_contiguous!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes,
            output_size, reduce_size, num_blocks, Val(2),
            ndrange=(block_size * num_blocks,),
        )
    else
        kernel! = _mapreduce_nd_by_block!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, neutral, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            output_size, reduce_size, num_blocks,
            ndrange=(block_size * num_blocks,),
        )
    end
end

# Smallest power-of-two >= n, capped at 256 (the original fixed block size) and
# floored at 32 (one warp) — used for the multigroup second pass, which only ever
# combines `reduce_groups` values (reduce_groups <= block_size by construction).
@inline function _pass2_block_size(n::Int)
    n <= 32  && return 32
    n <= 64  && return 64
    n <= 128 && return 128
    return 256
end


# CPU path

function _mapreduce_nd_cpu_sections!(
    f::F, op::OP, dst, src;
    init, neutral, max_tasks, min_elems,
) where {F, OP}
    Rother  = CartesianIndices(dst)
    Rreduce = _mapreduce_reduce_indices(src, dst)
    f_lanes, op_lanes = _lanefuncs(f, op, neutral)

    _foreachindex(eachindex(dst), HOST_BACKEND; max_tasks, min_elems) do idst
        @inbounds begin
            Iother = Rother[idst]
            res = neutral
            for Ireduce in Rreduce
                J = max(Iother, Ireduce)
                res = op_lanes(res, f_lanes(src[J]))
            end
            dst[idst] = _finish(op, init, dst, idst, res)
        end
    end
    dst
end
