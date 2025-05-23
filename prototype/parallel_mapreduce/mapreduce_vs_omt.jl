
import AcceleratedKernels as AK
import OhMyThreads as OMT
using BenchmarkTools


# Turns out we have the same performance as tmapreduce with just AK base threading
function mapreduce_omt(f, op, v; init)
    # MapReduce using OhMyThreads
    return OMT.tmapreduce(f, op, v; init=init)
end


function mapreduce_ak(f, op, v; init, max_tasks=Threads.nthreads())
    # MapReduce using AcceleratedKernels
    if max_tasks == 1
        return Base.mapreduce(f, op, v; init=init)
    end

    shared = Vector{typeof(init)}(undef, max_tasks)
    AK.itask_partition(length(v), max_tasks) do itask, irange
        @inbounds begin
            shared[itask] = Base.mapreduce(f, op, @view(v[irange]); init=init)
        end
    end
    return Base.reduce(op, shared; init=init)
end


v = rand(1_000_000)
f(x) = x^2
op(x, y) = x + y
init = eltype(v)(0)

@assert mapreduce_omt(f, op, v; init=init) == mapreduce_ak(f, op, v; init=init)

display(@benchmark mapreduce_omt(f, op, v; init=init))
display(@benchmark mapreduce_ak(f, op, v; init=init))
