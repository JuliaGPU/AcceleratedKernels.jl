# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels


# Nothing is exported, so that the functions can have Base's names without conflicts. The API is
# marked `public` instead, on Julia versions that know the keyword.
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("""
        public foreachindex, foraxes, map!, map,
               sort!, sort, sortperm!, sortperm, sort_by_key!,
               reduce, mapreduce, mapreducedim!, sum, prod, minimum, maximum, count,
               accumulate!, accumulate, cumsum, cumprod,
               searchsortedfirst!, searchsortedlast!, findall, reverse!, reverse, any, all,
               workspace, workspace_size, Workspace,
               Algorithm, Auto, CPUThreads,
               SortAlgorithm, MergeSort, RadixSort, BitonicSort,
               ReduceAlgorithm, BlockReduce, ScanAlgorithm, ScanPrefixes, SliceScan,
               FindallAlgorithm, ScanScatter, PredicateAlgorithm, ConcurrentWrite, ViaReduce,
               TaskPartitioner, task_partition, itask_partition,
               get_backend, synchronize
        """))
end


# Internal dependencies
using ArgCheck: @argcheck
import Adapt
using GPUArraysCore: @allowscalar, neutral_element
using KernelAbstractions
using KernelAbstractions: @context
# Sub-group operations, with KernelAbstractions 0.10
@static if isdefined(KernelAbstractions, :KernelInterface)
    const KI = KernelAbstractions.KernelInterface
end


# Exposed functions from upstream packages
const synchronize = KernelAbstractions.synchronize
const get_backend = KernelAbstractions.get_backend


# Include code from other files
include("utils.jl")
include("algorithms.jl")
include("workspace.jl")
include("task_partitioner.jl")
include("foreachindex.jl")
include("map.jl")
include("accumulator.jl")
include("block.jl")
include("searchsorted.jl")
include("sort/sort.jl")
include("reduce/reduce.jl")
include("accumulate/accumulate.jl")
include("findall.jl")
include("reverse.jl")
include("predicates.jl")


end     # module AcceleratedKernels
