# Reductions along `dims`. `mapreduce_nd!` merges adjacent dimensions of the source into segments
# (`merge_dims`), so that `dims=2` of a matrix or `dims=(1, 2)` of a 3-d array reduce one segment
# of elements a stride apart, and only reductions over non-adjacent dimensions decode an index per
# element. It then picks a launch shape from the sizes and strides of the outputs and of the
# reduction (`reduce_shape`) and launches its kernels (kernels_nd.jl). The shapes follow CUB's
# `DeviceSegmentedReduce`: a thread (`:by_thread`), a group of lanes (`:by_lanes`) or a block
# (`:by_block`, `:by_block_each`) per output, or several blocks per output in two passes
# (`:multigroup`); strided reductions into contiguous outputs give each block a tile of outputs
# (`:tiled_strided`, `:columns`), and reductions in element order have their own kernels
# (`:ordered`, `:columns_ordered`).

# Main entry point

# Reduce `src` into `dst`, which has `src`'s number of dimensions and size 1 along the reduced
# dimensions `rdims`; `A` is the accumulator type, and `init` is a value, `NoInit()` or
# `FoldInto()` (see `finish`). Without `commutative`, the GPU kernels combine elements in order.
function mapreduce_nd!(
    f::F, op::OP, dst::AbstractArray, src::MapReduceSource, backend::Backend,
    alg::Union{BlockReduce, CPUThreads.Partitioned}, ::Type{A};
    init,
    seed,
    rdims::Tuple,
    bufs::NamedTuple,
    commutative::Bool=true,
) where {F, OP, A}
    # Launch settings of the elementwise passes, and of the reduction kernels
    launch = alg isa BlockReduce ? (; block_size=alg.block_size) :
                                   (; max_tasks=alg.max_tasks, min_elems=alg.min_elems)

    # (the blocks that fill the device, which the shapes of few outputs aim to launch)
    tuning = reduce_tuning(backend, A)
    target_blocks = tuning.target_blocks

    src_sizes = size(src)

    # Degenerate cases begin; order of priority matters

    len = Base.prod(src_sizes[d] for d in rdims; init=1)

    # Empty reductions, e.g. reduce(+, rand(3, 0, 5), dims=2): `init`, else nothing is written
    if len == 0
        init isa Union{NoInit, FoldInto} || fill_on!(dst, init, backend)
        return dst
    end

    # Nothing to write, e.g. reduce(+, rand(3, 0, 5), dims=3), a 3×0×1 array
    isempty(dst) && return dst
    # (overwriting with single elements combines nothing, even when `op` always throws)
    len == 1 && init isa NoInit || check_combinable(op, f, A)

    # Every output reduces one element, e.g. reduce(+, rand(3, 5), dims=3): no reduced
    # dimensions, or only ones of size 1
    if len == 1
        # (a runtime dispatch on the GPU, see `launch_shape!`)
        single! = alg isa BlockReduce ? Base.inferencebarrier(mapreduce_singles!) :
                                        mapreduce_singles!
        single!(f, op, dst, src, backend, A; init, launch...)
        return dst
    end

    # Degenerate cases end

    dst_size = length(dst)

    if alg isa CPUThreads.Partitioned
        mapreduce_nd_threads!(f, op, dst, src; init, seed, launch...)
        return dst
    end
    block_size = alg.block_size

    # A single output of a contiguous source: the whole-array kernels, finishing on the device
    if dst_size == 1
        linear = linear_source(src, rdims)
        if linear !== nothing
            Base.inferencebarrier(mapreduce_1d_gpu)(
                f, op, linear, backend; init, seed, block_size,
                items_per_thread=alg.items_per_thread, max_blocks=target_blocks,
                partials=get(bufs, :partials, nothing), switch_below=0, commutative, dst)
            return dst
        end
    end

    # The kernels below index a flat buffer at `buffer[base_offset + Σ coordᵈ·strideᵈ + 1]`
    # (see `source_layout`)
    strided = strided_layout(src)
    layout = isnothing(strided) ? linear_layout(src) : strided
    buffer, base_offset = layout.buffer, layout.offset
    segs = merge_dims(src_sizes, layout.strides, rdims)
    reduce_size = len
    shape, reduce_groups = reduce_shape(dst_size, reduce_size, segs, block_size, tuning, A)
    commutative || ((shape, reduce_groups) = ordered_shape(shape, reduce_groups, dst_size,
                                                           reduce_size, block_size))

    # One output per thread of a source without a dense buffer: iterating over the Cartesian
    # indices of each output's elements saves decoding every linear index
    if isnothing(strided) && shape === :by_thread
        blocks = cld(dst_size, block_size)
        kernel! = by_thread_cartesian_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            src, dst, f, op, seed, init,
            CartesianIndices(dst), slice_indices(src, dst), dst_size,
            ndrange=(block_size * blocks,),
        )
        return dst
    end

    with_segments(segs) do reduce_strides, reduce_sizes, outer_strides, outer_sizes
        Base.inferencebarrier(launch_shape!)(
            Val(shape), reduce_groups, f, op, dst, buffer, backend, block_size, target_blocks;
            init, seed, bufs, base_offset, reduce_strides, reduce_sizes, outer_strides,
            outer_sizes, dst_size, reduce_size, dense_buffer=!isnothing(strided),
            items_per_thread=alg.items_per_thread,
        )
    end

    return dst
end

# Reductions whose every output reduces a single element, a partial result of the accumulator
# type `A`: `dst` and `src` have the same length.
function mapreduce_singles!(f::F, op::OP, dst, src, backend, ::Type{A}; init,
                            launch...) where {F, OP, A}
    foreachindex_on(eachindex(dst), backend; launch...) do i
        x = Base.mapreduce_first(f, op, src[i])
        dst[i] = finish(op, init, dst, i, A === Union{} ? x : convert(A, x))
    end
end

# Launch the kernels of a strided reduction of shape `shape`, with the shape's parameter
# `reduce_groups` (see `reduce_shape`).
#
# This and the other GPU launches of a reduction are called through `Base.inferencebarrier`.
# Inferring a launch through KernelAbstractions and the back-end takes far longer than the launch,
# and without the barrier a reduction with a new `f`, `op` or element type infers every launch it
# could make: all shapes, segment layouts and vector widths. A dynamic dispatch costs a fraction of
# a microsecond.
#
# `dense_buffer` is whether `buffer` is the dense buffer of the source, which the by-block kernel
# can read with vector loads, or a linearly indexed lazy source (see `source_layout`).
function launch_shape!(
    ::Val{shape}, reduce_groups, f::F, op::OP, dst, buffer, backend, block_size, target_blocks;
    init, seed, bufs, base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes,
    dst_size, reduce_size, dense_buffer, items_per_thread,
) where {shape, F, OP}
    if shape === :ordered
        partials = reduce_groups > 1 ? bufs.partials : nothing
        launch_ordered!(
            backend, block_size, items_per_thread, buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, reduce_groups, partials; dense_buffer,
        )
    elseif shape === :columns || shape === :columns_ordered
        ordered = shape === :columns_ordered
        groups = reduce_groups
        # (runs of `outer_sizes[1]` contiguous outputs, at the other kept dimensions' offsets)
        run = outer_sizes[1]
        runs = Base.tail(outer_strides), Base.tail(outer_sizes)
        cols = columns_outputs(run)
        blocks = cld(run, cols) * (dst_size ÷ run) * groups
        threads = min(block_size, COLUMNS_MAX_THREADS)
        kernel! = columns_kernel!(backend, threads)
        if groups == 1
            kernel!(buffer, dst, f, op, seed, init, base_offset, reduce_strides[1], dst_size,
                    run, dst_size ÷ run, runs..., reduce_size, 1, Val(cols), Val(ordered);
                    ndrange=(threads * blocks,))
        else
            # (the partial results of each output's groups, combined by a second pass)
            partial = bufs.partials
            kernel!(buffer, partial, f, op, seed, Unfinished(), base_offset,
                    reduce_strides[1], dst_size, run, dst_size ÷ run, runs..., reduce_size, groups,
                    Val(cols), Val(ordered);
                    ndrange=(threads * blocks,))
            pass2_block_size = combine_block_size(groups)
            if ordered
                ordered_pass!(backend, pass2_block_size, 1, partial, dst, PartialInputs(),
                              op, seed, init, 0, (1,), (dst_size,), (dst_size,),
                              (groups,), dst_size, groups, 1)
            else
                kernel2! = combine_groups_kernel!(backend, pass2_block_size)
                kernel2!(partial, dst, op, seed, init, dst_size, groups,
                         ndrange=(pass2_block_size * dst_size,))
            end
        end
    elseif shape === :by_lanes
        lanes = reduce_groups
        blocks = cld(dst_size, block_size ÷ lanes)
        kernel! = by_lanes_kernel!(backend, block_size)
        kernel!(
            buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, dst_size, reduce_size, Val(lanes),
            ndrange=(block_size * blocks,),
        )
    elseif shape === :tiled_strided
        rows_per_block = TILED_OUTPUTS
        blocks = cld(dst_size, rows_per_block)
        kernel! = tiled_strided_kernel!(backend, block_size)
        kernel!(
            buffer, dst, f, op, seed, init,
            base_offset, reduce_strides[1], dst_size, reduce_size, Val(rows_per_block),
            ndrange=(block_size * blocks,),
        )
    elseif shape === :by_thread
        blocks = cld(dst_size, block_size)
        kernel! = by_thread_kernel!(backend, block_size)
        kernel!(
            buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    elseif shape === :multigroup
        partial = bufs.partials
        kernel! = multigroup_kernel!(backend, block_size)
        kernel!(
            buffer, partial, f, op, seed,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            Val(dst_size), reduce_size, reduce_groups,
            ndrange=(block_size * dst_size * reduce_groups,),
        )

        pass2_block_size = combine_block_size(reduce_groups)
        kernel2! = combine_groups_kernel!(backend, pass2_block_size)
        kernel2!(
            partial, dst, op, seed, init,
            dst_size, reduce_groups,
            ndrange=(pass2_block_size * dst_size,),
        )
    else
        # (`:by_block_each` launches a block for each of its few outputs)
        launch_blocks = shape === :by_block ? min(dst_size, target_blocks) : dst_size
        launch_by_block!(
            backend, block_size, buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, launch_blocks, dense_buffer,
        )
    end

    return nothing
end

# Launch the by-block kernel with `num_blocks` blocks, each reducing the outputs `iblock`,
# `iblock + num_blocks`, ...; with vector loads where `buffer` is a dense buffer they suit
function launch_by_block!(
    backend, block_size, buffer, dst, f::F, op::OP, seed, init,
    base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
    output_size, reduce_size, num_blocks, dense_buffer,
) where {F, OP}
    W = dense_buffer ? vector_width(
        buffer, base_offset, outer_strides, reduce_strides, reduce_size, block_size,
    ) : 0
    if W > 0
        kernel! = by_block_vector_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes,
            output_size, reduce_size, num_blocks, Val(W),
            ndrange=(block_size * num_blocks,),
        )
    else
        kernel! = by_block_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            output_size, reduce_size, num_blocks,
            ndrange=(block_size * num_blocks,),
        )
    end
end

# The block size of a second pass that combines `n` partial results per output: the next power
# of two, from 32 to 256
@inline function combine_block_size(n::Int)
    n <= 32  && return 32
    n <= 64  && return 64
    n <= 128 && return 128
    return 256
end


# CPU path

# Reduce `src` into `dst` on Julia threads, one output after the other in each task
function mapreduce_nd_threads!(
    f::F, op::OP, dst, src;
    init, seed, max_tasks, min_elems,
) where {F, OP}
    Rother  = CartesianIndices(dst)
    Rreduce = slice_indices(src, dst)
    mapf, accop = seeded_ops(f, op, seed)

    foreachindex_on(eachindex(dst), HOST_BACKEND; max_tasks, min_elems) do idst
        @inbounds begin
            Iother = Rother[idst]
            res = seed
            for Ireduce in Rreduce
                J = max(Iother, Ireduce)
                res = accop(res, mapf(src[J]))
            end
            dst[idst] = finish(op, init, dst, idst, res)
        end
    end
    dst
end
