# File   : AcceleratedKernels.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module AcceleratedKernels


# No functions exported by default, to allow having the same names as Base without conflicts


# Internal dependencies
using ArgCheck: @argcheck
using GPUArrays: GPUArrays, AbstractGPUVector, AbstractGPUArray, @allowscalar
using KernelAbstractions
using Polyester: @batch
import OhMyThreads as OMT
using Unrolled: @unroll, unrolled_map, FixedRange


# Exposed functions from upstream packages
const synchronize = KernelAbstractions.synchronize
const get_backend = KernelAbstractions.get_backend
const neutral_element = GPUArrays.neutral_element


# Include code from other files
include("utils.jl")
include("task_partitioner.jl")
include("foreachindex.jl")
include("map.jl")
include("sort/sort.jl")
include("reduce/reduce.jl")
include("accumulate/accumulate.jl")
include("searchsorted.jl")
include("predicates.jl")
include("arithmetics.jl")


end     # module AcceleratedKernels
