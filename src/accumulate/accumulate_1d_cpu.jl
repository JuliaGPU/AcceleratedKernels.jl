function accumulate_1d_cpu!(
    op, v::AbstractArray, backend::Backend, alg::CPUThreads.Partitioned;
    init,
    seed,
    inclusive::Bool,
)
    # Trivial case
    if length(v) == 0
        return v
    end

    # For exclusive accumulation, each task section / chunk must have at least 2 elements to be
    # correct (otherwise we have to include more complicated logic in the threaded code); the
    # resolution checks `min_elems`
    max_tasks, min_elems = alg.max_tasks, alg.min_elems

    start = scan_start(v, init, seed)
    _, op = seeded_ops(PartialInputs(), op, seed)

    if task_count(length(v), max_tasks, min_elems) == 1
        serial_scan!(op, v; start, seed, inclusive)
        return v
    end

    # Scan each task's section with the requested inclusivity, seeding only the first one with
    # `start`, and save each section's total
    tp = TaskPartitioner(length(v), max_tasks, min_elems)
    shared = Vector{typeof(seed)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        shared[itask] = serial_scan!(
            op, @view(v[irange]);
            start=itask == 1 ? start : seed,
            seed,
            inclusive,
        )
    end

    # Now accumulate the totals of each task; the number of tasks is small enough (even for
    # 144-thread HPC nodes) that there is no need to do decoupled lookbacks
    serial_scan!(op, shared; start=seed, seed, inclusive=true)

    # Now prepend the running total of all previous tasks to each element, except in the first
    # task. An exclusive section starts from `seed`, which may be an unset `Optional` that cannot
    # be stored: its first element is the running total itself.
    itask_partition(tp) do itask, irange
        @inbounds begin
            if itask != 1
                carry = shared[itask - 1]
                for i in irange
                    v[i] = if !inclusive && i == first(irange)
                        lower(eltype(v), carry)
                    else
                        lower(eltype(v), op(carry, lift(seed, v[i])))
                    end
                end
            end
        end
    end

    return v
end


# Scan a section sequentially from `start`, returning its total. `op` combines partial results
# (`Optional`s when `seed` is an unset one); elements are lifted to them and stored back lowered.
function serial_scan!(op, v; start, seed, inclusive)
    @inbounds begin
        running = start
        for i in eachindex(v)
            x = lift(seed, v[i])
            if inclusive
                running = op(running, x)
                store_partial!(v, i, running)
            else
                store_partial!(v, i, running)
                running = op(running, x)
            end
        end
    end
    return running
end
