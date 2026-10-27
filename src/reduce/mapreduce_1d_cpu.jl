# Reduce the non-empty `src` to a value on Julia threads; `seed` and `init` as for
# `mapreduce_1d_gpu`.
function mapreduce_1d_cpu(
    f::F, op::OP, src::MapReduceSource, backend::Backend;
    init,
    seed,
    max_tasks::Int,
    min_elems::Int,
) where {F, OP}
    f, accop = seeded_ops(f, op, seed)
    # (a `Broadcasted` object has no views; one too long for `LinearBroadcast` is reduced serially)
    src isa Base.Broadcast.Broadcasted && (src = linear_view(src))
    if src isa Base.Broadcast.Broadcasted || task_count(length(src), max_tasks, min_elems) == 1
        return finish(op, init, nothing, 0, Base.mapreduce(f, accop, src; init=seed))
    end
    tp = TaskPartitioner(length(src), max_tasks, min_elems)

    # Each task reduces an independent chunk of the array
    shared = Vector{typeof(seed)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        @inbounds begin
            # This shared buffer is only modified once per task, so false sharing is not a problem
            shared[itask] = Base.mapreduce(f, accop, @view(src[irange]); init=seed)
        end
    end
    return finish(op, init, nothing, 0, Base.reduce(accop, shared; init=seed))
end
