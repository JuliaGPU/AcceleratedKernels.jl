# Reductions along `dims`. Their setup merges adjacent dimensions of the source into segments
# (`merge_dims`), so that `dims=2` of a matrix or `dims=(1, 2)` of a 3-d array reduce one segment
# of elements a stride apart, and only reductions over non-adjacent dimensions decode an index per
# element. It then picks a launch shape from the sizes and strides of the outputs and of the
# reduction (`select_launch`, shapes.jl), which `mapreduce_nd!` launches (`launch!`, with the
# kernels of kernels_nd.jl).

# Main entry point

# Reduce `src` into `dst`, which has `src`'s number of dimensions and size 1 along the reduced
# dimensions `rdims`; `A` is the accumulator type, `init` is a value, `NoInit()` or `FoldInto()`
# (see `finish`), and `launch` is the setup's (see `select_launch`).
function mapreduce_nd!(
    f::F, op::OP, dst::AbstractArray, src::MapReduceSource, backend::Backend,
    alg::Union{BlockReduce, CPUThreads.Partitioned}, ::Type{A};
    init,
    seed,
    rdims::Tuple,
    bufs::NamedTuple,
    launch,
) where {F, OP, A}
    # Launch settings of the elementwise passes, and of the reduction kernels
    launch_kw = alg isa BlockReduce ? (; block_size=alg.block_size) :
                                      (; max_tasks=alg.max_tasks, min_elems=alg.min_elems)

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
        # (a runtime dispatch on the GPU, see `launch!`)
        single! = alg isa BlockReduce ? Base.inferencebarrier(mapreduce_singles!) :
                                        mapreduce_singles!
        single!(f, op, dst, src, backend, A; init, launch_kw...)
        return dst
    end

    # Degenerate cases end

    dst_size = length(dst)

    if alg isa CPUThreads.Partitioned
        mapreduce_nd_threads!(f, op, dst, src; init, seed, launch_kw...)
        return dst
    end
    shape = launch.shape

    # A single output of a contiguous source: the whole-array kernels, finishing on the device
    if shape isa Whole
        Base.inferencebarrier(mapreduce_1d_gpu)(
            f, op, linear_source(src, launch.buffer, launch.offset), backend; init, seed,
            block_size=alg.block_size, items_per_thread=alg.items_per_thread,
            max_blocks=reduce_tuning(backend, A).target_blocks,
            partials=get(bufs, :partials, nothing), switch_below=0,
            commutative=!shape.ordered, dst)
        return dst
    end

    # (the kernels below index a flat buffer at `buffer[offset + Σ coordᵈ·strideᵈ + 1]`)
    buffer, dense = kernel_buffer(src, launch.buffer)
    with_segments(launch.segs) do reduce_strides, reduce_sizes, outer_strides, outer_sizes
        ix = (; base_offset=launch.offset, reduce_strides, reduce_sizes, outer_strides,
              outer_sizes, dst_size, reduce_size=len, dense)
        Base.inferencebarrier(launch!)(shape, f, op, dst, src, buffer, backend, alg, bufs, init,
                                       seed, ix)
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

# Launch the kernels of a reduction along `dims` of shape `shape` into `dst`, reading `buffer`
# (see `kernel_buffer`) with the indices `ix`: the offsets of the segments `reduce_*` and `outer_*`
# from `base_offset`, the sizes, and whether `buffer` is the source's dense buffer (`dense`), which
# some kernels read with vector loads.
#
# This and the other GPU launches of a reduction are called through `Base.inferencebarrier`.
# Inferring a launch through KernelAbstractions and the back-end takes far longer than the launch,
# and without the barrier a reduction with a new `f`, `op` or element type infers every launch it
# could make: all shapes, segment layouts and vector widths. A dynamic dispatch costs a fraction of
# a microsecond.
function launch!(::ByThread, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes, dst_size,
       reduce_size, dense) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, block_size)
    if dense
        by_thread_kernel!(backend, block_size)(
            buffer, dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    else
        # (iterating over the Cartesian indices of each output's elements saves decoding every
        # linear index)
        by_thread_cartesian_kernel!(backend, block_size)(
            src, dst, f, op, seed, init,
            CartesianIndices(dst), slice_indices(src, dst), dst_size,
            ndrange=(block_size * blocks,),
        )
    end
end

function launch!(shape::ByLanes, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, dst_size, reduce_size) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, block_size ÷ shape.lanes)
    by_lanes_kernel!(backend, block_size)(
        buffer, dst, f, op, seed, init,
        base_offset, outer_strides, outer_sizes, dst_size, reduce_size, Val(shape.lanes),
        ndrange=(block_size * blocks,),
    )
end

# (with vector loads where `buffer` is a dense buffer they suit)
function launch!(shape::ByBlock, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes, dst_size,
       reduce_size, dense) = ix
    block_size, blocks = alg.block_size, shape.blocks
    W = dense ? vector_width(buffer, base_offset, outer_strides, reduce_strides, reduce_size,
                             block_size) : 0
    if W > 0
        kernel! = by_block_vector_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init, base_offset, outer_strides, outer_sizes,
            dst_size, reduce_size, blocks, Val(W),
            ndrange=(block_size * blocks,),
        )
    else
        kernel! = by_block_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init, base_offset, outer_strides, outer_sizes,
            reduce_strides, reduce_sizes, dst_size, reduce_size, blocks,
            ndrange=(block_size * blocks,),
        )
    end
end

function launch!(shape::ByBlocks, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes, dst_size,
       reduce_size) = ix
    block_size, groups = alg.block_size, shape.groups
    partials = bufs.partials
    multigroup_kernel!(backend, block_size)(
        buffer, partials, f, op, seed,
        base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
        Val(dst_size), reduce_size, groups,
        ndrange=(block_size * dst_size * groups,),
    )
    combine_groups!(backend, partials, dst, op, seed, init, dst_size, groups, false)
end

function launch!(shape::Columns, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, outer_strides, outer_sizes, dst_size, reduce_size) = ix
    (; outputs, groups, ordered) = shape
    # (runs of `outer_sizes[1]` contiguous outputs, at the other kept dimensions' offsets)
    run = outer_sizes[1]
    runs = Base.tail(outer_strides), Base.tail(outer_sizes)
    blocks = cld(run, outputs) * (dst_size ÷ run) * groups
    threads = min(alg.block_size, COLUMNS_MAX_THREADS)
    # (with several groups, the partial results of each output's groups, for a second pass)
    out, out_init = groups == 1 ? (dst, init) : (bufs.partials, Unfinished())
    columns_kernel!(backend, threads)(
        buffer, out, f, op, seed, out_init, base_offset, reduce_strides[1], dst_size,
        run, dst_size ÷ run, runs..., reduce_size, groups, Val(outputs), Val(ordered);
        ndrange=(threads * blocks,),
    )
    groups == 1 || combine_groups!(backend, out, dst, op, seed, init, dst_size, groups, ordered)
end

function launch!(::Tiled, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, dst_size, reduce_size) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, TILED_OUTPUTS)
    tiled_strided_kernel!(backend, block_size)(
        buffer, dst, f, op, seed, init,
        base_offset, reduce_strides[1], dst_size, reduce_size, Val(TILED_OUTPUTS),
        ndrange=(block_size * blocks,),
    )
end

function launch!(shape::Ordered, f::F, op::OP, dst, src, buffer, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes, dst_size,
       reduce_size, dense) = ix
    groups = shape.groups
    launch_ordered!(
        backend, alg.block_size, alg.items_per_thread, buffer, dst, f, op, seed, init,
        base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
        dst_size, reduce_size, groups, groups > 1 ? bufs.partials : nothing;
        dense_buffer=dense,
    )
end

# The second pass of a reduction split across `groups` blocks per output: a block per output
# combines the partial results `partials` of its groups (`dst_size` apart) into `dst`, in order if
# `ordered`
function combine_groups!(backend, partials, dst, op::OP, seed, init, dst_size, groups,
                         ordered) where {OP}
    block_size = combine_block_size(groups)
    if ordered
        ordered_pass!(backend, block_size, 1, partials, dst, PartialInputs(), op, seed, init,
                      0, (1,), (dst_size,), (dst_size,), (groups,), dst_size, groups, 1)
    else
        by_block_kernel!(backend, block_size)(
            partials, dst, PartialInputs(), op, seed, init, 0, (1,), (dst_size,), (dst_size,),
            (groups,), dst_size, groups, dst_size, ndrange=(block_size * dst_size,))
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
