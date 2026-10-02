# Reduce the non-empty `src` to a value on Julia threads; `neutral` and `init` as for
# `mapreduce_1d_gpu`.
function mapreduce_1d_cpu(
    f::F, op::OP, src::MapReduceSource, backend::Backend;
    init,
    neutral,
    max_tasks::Int,
    min_elems::Int,
) where {F, OP}
    f, op_lanes = _lanefuncs(f, op, neutral)
    tp = TaskPartitioner(length(src), max_tasks, min_elems)
    if tp.num_tasks == 1
        return _finish(op, init, nothing, 0, Base.mapreduce(f, op_lanes, src; init=neutral))
    end

    # Each task reduces an independent chunk of the array
    shared = Vector{typeof(neutral)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        @inbounds begin
            # This shared buffer is only modified once per task, so false sharing is not a problem
            shared[itask] = _mapreduce_chunk(f, op_lanes, src, irange, neutral)
        end
    end
    return _finish(op, init, nothing, 0, Base.reduce(op_lanes, shared; init=neutral))
end

# Reduce the elements of `src` at the linear indices `irange`; a `Broadcasted` object has no
# views, so its elements are computed by index
_mapreduce_chunk(f::F, op::OP, src::AbstractArray, irange, neutral) where {F, OP} =
    Base.mapreduce(f, op, @view(src[irange]); init=neutral)
_mapreduce_chunk(f::F, op::OP, src::Base.Broadcast.Broadcasted, irange, neutral) where {F, OP} =
    Base.mapreduce(i -> f(@inbounds(src[i])), op, irange; init=neutral)
