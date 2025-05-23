function mapreduce_1d(
    f, op, src::AbstractArray, backend::CPU;
    init,
    neutral=neutral_element(op, eltype(src)),

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # GPU settings - ignored here
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if max_tasks == 1
        return op(init, Base.mapreduce(f, op, src; init=neutral))
    end

    tp = TaskPartitioner(length(src), max_tasks, min_elems)
    if tp.num_tasks == 1
        return op(init, Base.mapreduce(f, op, src; init=neutral))
    end

    # Each task reduces an independent chunk of the array
    shared = Vector{typeof(init)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        @inbounds begin
            # This shared buffer is only modified once per task, so false sharing is not a problem
            shared[itask] = Base.mapreduce(f, op, @view(src[irange]); init=neutral)
        end
    end
    return op(init, Base.reduce(op, shared; init=neutral))
end
