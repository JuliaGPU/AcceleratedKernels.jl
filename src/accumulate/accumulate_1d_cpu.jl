function accumulate_1d!(
    op, v::AbstractArray, backend::CPU, alg;
    init,
    neutral,
    inclusive::Bool=true,

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=2,

    # GPU settings - not used
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
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
        return _accumulate_1d_cpu_section!(op, v; init, inclusive)
    end

    # Save each task's final accumulated value
    shared = Vector{eltype(v)}(undef, tp.num_tasks)
    itask_partition(tp) do itask, irange
        @inbounds begin
            if itask == 1
                _accumulate_1d_cpu_section!(
                    op, @view(v[irange]);
                    init=init,
                    inclusive=inclusive,
                )
            else
                # Later sections should always be inclusively accumulated
                _accumulate_1d_cpu_section!(
                    op, @view(v[irange]);
                    init=neutral,
                    inclusive=true,
                )
            end
            shared[itask] = v[irange.stop]
        end
    end

    # Now accumulate the final values of each task; the number of tasks is small enough (even for
    # 144-thread HPC nodes) that there is no need to do decoupled lookbacks
    _accumulate_1d_cpu_section!(op, shared; init=neutral, inclusive=true)

    # Now add the final values of each task, except the first one
    itask_partition(tp) do itask, irange
        @inbounds begin
            if itask != 1
                for i in irange
                    v[i] = op(v[i], shared[itask - 1])
                end
            end
        end
    end

    return v
end


function _accumulate_1d_cpu_section!(op, v; init, inclusive)
    @inbounds begin
        if inclusive
            running = init
            for i in eachindex(v)
                running = op(running, v[i])
                v[i] = running
            end
        else
            running = init
            for i in eachindex(v)
                v[i], running = running, op(running, v[i])
            end
        end
    end
    return v
end
