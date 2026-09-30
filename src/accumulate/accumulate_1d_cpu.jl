function accumulate_1d_cpu!(
    op, v::AbstractArray, backend::Backend, alg::CPUThreads.Partitioned;
    init,
    neutral,
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

    seed = _scan_first_seed(v, init, neutral)
    _, op = _lanefuncs(_Partials(), op, neutral)

    # First accumulate chunks independently
    tp = TaskPartitioner(length(v), max_tasks, min_elems)
    if tp.num_tasks == 1
        _accumulate_1d_cpu_section!(op, v; seed, neutral, inclusive)
        return v
    end

    # Scan each task's section with the requested inclusivity, seeding only the first one with
    # `seed`, and save each section's total
    shared = Vector{typeof(neutral)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        shared[itask] = _accumulate_1d_cpu_section!(
            op, @view(v[irange]);
            seed=itask == 1 ? seed : neutral,
            neutral,
            inclusive,
        )
    end

    # Now accumulate the totals of each task; the number of tasks is small enough (even for
    # 144-thread HPC nodes) that there is no need to do decoupled lookbacks
    _accumulate_1d_cpu_section!(op, shared; seed=neutral, neutral, inclusive=true)

    # Now prepend the running total of all previous tasks to each element, except in the first
    # task. An exclusive section starts with its seed, `neutral`, which may be an empty lane that
    # cannot be stored: its first element is the running total itself.
    itask_partition(tp) do itask, irange
        @inbounds begin
            if itask != 1
                carry = shared[itask - 1]
                for i in irange
                    v[i] = if !inclusive && i == first(irange)
                        _lower(eltype(v), carry)
                    else
                        _lower(eltype(v), op(carry, _lift(neutral, v[i])))
                    end
                end
            end
        end
    end

    return v
end


# Scan a section sequentially from `seed`, returning its total. `op` combines partial results
# (lanes when `neutral` is an empty lane); elements are lifted to them and stored back lowered.
function _accumulate_1d_cpu_section!(op, v; seed, neutral, inclusive)
    @inbounds begin
        running = seed
        for i in eachindex(v)
            x = _lift(neutral, v[i])
            if inclusive
                running = op(running, x)
                _store!(v, i, running)
            else
                _store!(v, i, running)
                running = op(running, x)
            end
        end
    end
    return running
end
