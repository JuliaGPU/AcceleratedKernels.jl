import AcceleratedKernels as AK
using KernelAbstractions
using Statistics
using Test
using Random
import Pkg

# Set to true when testing backends that support this
const TEST_DL = Ref{Bool}(false)

# Pass command-line argument to test suite to install the right backend, e.g.
#   julia> import Pkg
#   julia> Pkg.test(test_args=["--oneAPI"])
if "--CUDA" in ARGS
    Pkg.add("CUDA")
    using CUDA
    CUDA.versioninfo()
    const BACKEND = CUDABackend()
    TEST_DL[] = true
elseif "--oneAPI" in ARGS
    Pkg.add("oneAPI")
    using oneAPI
    oneAPI.versioninfo()
    const BACKEND = oneAPIBackend()

    # FIXME: need atomic orderings for `DecoupledLookback` in oneAPI
    # TEST_DL[] = true
elseif "--AMDGPU" in ARGS
    Pkg.add("AMDGPU")
    using AMDGPU
    AMDGPU.versioninfo()
    const BACKEND = ROCBackend()
    TEST_DL[] = true
elseif "--Metal" in ARGS
    Pkg.add("Metal")
    using Metal
    Metal.versioninfo()
    const BACKEND = MetalBackend()
elseif "--OpenCL" in ARGS
    Pkg.add(name="OpenCL", rev="master")
    Pkg.add(name="SPIRVIntrinsics", rev="master")
    Pkg.add("pocl_jll")
    using pocl_jll
    using OpenCL
    OpenCL.versioninfo()
    const BACKEND = OpenCLBackend()
elseif !@isdefined(BACKEND)
    # Otherwise do CPU tests
    using InteractiveUtils
    InteractiveUtils.versioninfo()
    const BACKEND = get_backend([])
end

const IS_CPU_BACKEND = BACKEND == get_backend([])

global prefer_threads::Bool = !(IS_CPU_BACKEND && "--cpuKA" in ARGS)

array_from_host(h_arr::AbstractArray, dtype=nothing) = array_from_host(BACKEND, h_arr, dtype)
function array_from_host(backend, h_arr::AbstractArray, dtype=nothing)
    d_arr = KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
    copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))      # Allow unmaterialised types, e.g. ranges
    d_arr
end

@testset "Aqua" begin
    using Aqua
    Aqua.test_all(AK)
end

include("statistics.jl")
include("partition.jl")
include("looping.jl")
include("map.jl")
include("sort.jl")
include("reduce.jl")
include("accumulate.jl")
include("predicates.jl")
include("binarysearch.jl")
