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
        mapreduce_slices!(f, op, dst, src, HOST_BACKEND; init, seed, launch_kw...)
        return dst
    end
    shape = launch.shape

    # (the kernels index a flat buffer at `buffer[offset + Σ coordᵈ·strideᵈ + 1]`; the launches
    # take its kind as a type and build it themselves, so that inference of a launch for a new type
    # of source visits only the buffer that launch reads)
    kind = buffer_val(launch.buffer)

    # A single output of a contiguous source: the whole-array kernels, finishing on the device
    if shape isa Whole
        Base.inferencebarrier(mapreduce_1d_gpu)(
            f, op, Base.inferencebarrier(linear_source)(src, kind, launch.offset),
            backend; init, seed,
            block_size=alg.block_size, items_per_thread=alg.items_per_thread,
            max_blocks=tuning(reduce_tuning, backend, A).target_blocks,
            partials=get(bufs, :partials, nothing), switch_below=0,
            commutative=!shape.ordered, dst)
        return dst
    end

    Base.inferencebarrier(launch!)(shape, f, op, dst, src, kind, backend, alg, bufs, init, seed,
                                   segment_indices(launch, dst_size, len))
    return dst
end

# The indices `launch!` takes, from the setup's `launch` alone: inference of a function that also
# takes the source is repeated for every new type of source, and grows with the size of that type
segment_indices(launch, dst_size, len) =
    with_segments(launch.segs) do reduce_strides, reduce_sizes, outer_strides, outer_sizes
        (; base_offset=launch.offset, reduce_strides, reduce_sizes, outer_strides,
         outer_sizes, dst_size, reduce_size=len)
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

# Launch the kernels of a reduction along `dims` of shape `shape` of `src` into `dst`, reading the
# buffer of kind `kind` (see `kernel_buffer`) with the indices `ix`: the offsets of the segments
# `reduce_*` and `outer_*` from `base_offset`, and the sizes. Some kernels read a dense buffer
# (`isdense_buffer`) with vector loads.
#
# This and the other GPU launches of a reduction are called through `Base.inferencebarrier`.
# Inferring a launch through KernelAbstractions and the back-end takes far longer than the launch,
# and without the barrier a reduction with a new `f`, `op` or element type infers every launch it
# could make: all shapes, segment layouts and vector widths. A dynamic dispatch costs a fraction of
# a microsecond.
function launch!(::ByThread, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes, dst_size,
       reduce_size) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, block_size)
    if isdense_buffer(kind)
        by_thread_kernel!(backend, block_size)(
            kernel_buffer(src, kind), dst, f, op, seed, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    else
        mapreduce_slices!(f, op, dst, src, backend; init, seed, block_size)
    end
end

function launch!(shape::ByLanes, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, dst_size, reduce_size) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, block_size ÷ shape.lanes)
    by_lanes_kernel!(backend, block_size)(
        kernel_buffer(src, kind), dst, f, op, seed, init,
        base_offset, outer_strides, outer_sizes, dst_size, reduce_size, Val(shape.lanes),
        reduce_subgroup_width(backend, typeof(seed), block_size),
        ndrange=(block_size * blocks,),
    )
end

# (with vector loads where `buffer` is a dense buffer they suit)
function launch!(shape::ByBlock, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes, dst_size,
       reduce_size) = ix
    buffer = kernel_buffer(src, kind)
    block_size, blocks = alg.block_size, shape.blocks
    W = isdense_buffer(kind) ? vector_width(buffer, base_offset, outer_strides, reduce_strides,
                                            reduce_size, block_size) : 0
    if W > 0
        kernel! = by_block_vector_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init, base_offset, outer_strides, outer_sizes,
            dst_size, reduce_size, blocks, Val(W),
            reduce_subgroup_width(backend, typeof(seed), block_size),
            ndrange=(block_size * blocks,),
        )
    else
        kernel! = by_block_kernel!(backend, block_size)
        Base.inferencebarrier(kernel!)(
            buffer, dst, f, op, seed, init, base_offset, outer_strides, outer_sizes,
            reduce_strides, reduce_sizes, dst_size, reduce_size, blocks,
            reduce_subgroup_width(backend, typeof(seed), block_size),
            ndrange=(block_size * blocks,),
        )
    end
end

function launch!(shape::ByBlocks, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes, dst_size,
       reduce_size) = ix
    block_size, groups = alg.block_size, shape.groups
    partials = bufs.partials
    multigroup_kernel!(backend, block_size)(
        kernel_buffer(src, kind), partials, f, op, seed,
        base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
        Val(dst_size), reduce_size, groups,
        reduce_subgroup_width(backend, typeof(seed), block_size),
        ndrange=(block_size * dst_size * groups,),
    )
    combine_groups!(backend, partials, dst, op, seed, init, dst_size, groups, false,
                    block_size)
end

function launch!(shape::Columns, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
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
        kernel_buffer(src, kind), out, f, op, seed, out_init, base_offset, reduce_strides[1],
        dst_size, run, dst_size ÷ run, runs..., reduce_size, groups, Val(outputs), Val(ordered);
        ndrange=(threads * blocks,),
    )
    groups == 1 || combine_groups!(backend, out, dst, op, seed, init, dst_size, groups, ordered,
                                   threads)
end

function launch!(::Tiled, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, dst_size, reduce_size) = ix
    block_size = alg.block_size
    blocks = cld(dst_size, TILED_OUTPUTS)
    tiled_strided_kernel!(backend, block_size)(
        kernel_buffer(src, kind), dst, f, op, seed, init,
        base_offset, reduce_strides[1], dst_size, reduce_size, Val(TILED_OUTPUTS),
        ndrange=(block_size * blocks,),
    )
end

function launch!(shape::Ordered, f::F, op::OP, dst, src, kind, backend, alg, bufs, init,
                 seed, ix) where {F, OP}
    (; base_offset, reduce_strides, reduce_sizes, outer_strides, outer_sizes, dst_size,
       reduce_size) = ix
    groups = shape.groups
    launch_ordered!(
        backend, alg.block_size, alg.items_per_thread, kernel_buffer(src, kind), dst, f, op,
        seed, init,
        base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
        dst_size, reduce_size, groups, groups > 1 ? bufs.partials : nothing;
        dense_buffer=isdense_buffer(kind),
    )
end

# The second pass of a reduction split across `groups` blocks per output: a block per output
# combines the partial results `partials` of its groups (`dst_size` apart) into `dst`, in order if
# `ordered`, with blocks of at most `max_block_size` threads (the first pass's, which fit local
# memory)
function combine_groups!(backend, partials, dst, op::OP, seed, init, dst_size, groups,
                         ordered, max_block_size) where {OP}
    block_size = min(combine_block_size(groups), max_block_size)
    if ordered
        ordered_pass!(backend, block_size, 1, partials, dst, PartialInputs(), op, seed, init,
                      0, (1,), (dst_size,), (dst_size,), (groups,), dst_size, groups, 1)
    else
        by_block_kernel!(backend, block_size)(
            partials, dst, PartialInputs(), op, seed, init, 0, (1,), (dst_size,), (dst_size,),
            (groups,), dst_size, groups, dst_size,
            reduce_subgroup_width(backend, typeof(seed), block_size),
            ndrange=(block_size * dst_size,))
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


# Reduce `src` into `dst` with a thread per output, which reduces its slice in turn: on Julia
# threads, or on the GPU for sources without a dense buffer (iterating over the Cartesian indices
# of each output's elements saves decoding every linear index)
function mapreduce_slices!(f::F, op::OP, dst, src, backend; init, seed, launch...) where {F, OP}
    Rother  = CartesianIndices(dst)
    Rreduce = slice_indices(src, dst)
    mapf, accop = seeded_ops(f, op, seed)

    foreachindex_on(eachindex(dst), backend; launch...) do idst
        @inbounds begin
            Iother = Rother[idst]
            res = kernel_value(seed)
            for Ireduce in Rreduce
                J = max(Iother, Ireduce)
                res = accop(res, mapf(src[J]))
            end
            dst[idst] = finish(op, init, dst, idst, res)
        end
    end
    dst
end
