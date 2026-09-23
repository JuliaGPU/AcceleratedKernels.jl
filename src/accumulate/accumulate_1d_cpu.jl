function accumulate_1d_cpu!(
    op, v::AbstractArray, backend::Backend, alg;
    init,
    neutral,
    inclusive::Bool,

    # CPU settings
    max_tasks::Int,
    min_elems::Int,

    # GPU settings - not used
    block_size::Int,
    temp::Union{Nothing, AbstractArray},
    temp_flags::Union{Nothing, AbstractArray},
)
    # Trivial case
    if length(v) == 0
        return v
    end

    # Sanity checks - for exclusive accumulation, each task section / chunk must have at least 2
    # elements to be correct (otherwise we have to include more complicated logic in the threaded
    # code); it makes no sense to have each task accumulate only 1 element anyways
    @argcheck min_elems >= 2

    # First accumulate chunks independently
    tp = TaskPartitioner(length(v), max_tasks, min_elems)
    if tp.num_tasks == 1
        _accumulate_1d_cpu_section!(op, v; init, inclusive)
        return v
    end

    # Scan each task's section with the requested inclusivity, seeding only the first one with
    # `init`, and save each section's total
    shared = Vector{eltype(v)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        shared[itask] = _accumulate_1d_cpu_section!(
            op, @view(v[irange]);
            init=itask == 1 ? init : neutral,
            inclusive,
        )
    end

    # Now accumulate the totals of each task; the number of tasks is small enough (even for
    # 144-thread HPC nodes) that there is no need to do decoupled lookbacks
    _accumulate_1d_cpu_section!(op, shared; init=neutral, inclusive=true)

    # Now prepend the running total of all previous tasks to each element, except in the first task
    itask_partition(tp) do itask, irange
        @inbounds begin
            if itask != 1
                for i in irange
                    v[i] = op(shared[itask - 1], v[i])
                end
            end
        end
    end

    return v
end


# Scan a section sequentially, returning its total (including `init`).
function _accumulate_1d_cpu_section!(op, v; init, inclusive)
    @inbounds begin
        running = init
        if inclusive
            for i in eachindex(v)
                running = op(running, v[i])
                v[i] = running
            end
        else
            for i in eachindex(v)
                v[i], running = running, op(running, v[i])
            end
        end
    end
    return running
end
