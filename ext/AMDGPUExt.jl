module AMDGPUExt

using AMDGPU
import AcceleratedKernels as AK
using PrecompileTools: @setup_workload, @compile_workload


# AMDGPU's own workload can't cache these operations, as it doesn't use the GPU: ROCm is only
# discovered when AMDGPU is loaded, which it isn't while it precompiles, but is while this
# extension does. The arrays are just large enough to select the kernels that a vector of
# 10^6 elements and a 1000×1000 matrix do; at the default tunings, the matrix's columns are
# reduced by a block each, which takes longer columns than for CUDA. Without a GPU while
# precompiling, these operations are only compiled when first used. Like AMDGPU's own
# workload, this needs 1.12: before, GPU code compiled while precompiling leaks into the host
# code of the image.
@static if VERSION >= v"1.12-"

# `functional` only checks that ROCm's libraries were found, so also count the devices,
# without the checked wrapper, which throws when there are none. older versions of AMDGPU
# only have the checked one; skip the workload there.
function has_device()
    AMDGPU.functional() || return false
    isdefined(AMDGPU.HIP, :unchecked_hipGetDeviceCount) || return false
    count = Ref{Cint}(0)
    AMDGPU.HIP.unchecked_hipGetDeviceCount(count) == AMDGPU.HIP.hipSuccess || return false
    return count[] > 0
end

@setup_workload begin
    if has_device()
        @compile_workload begin
            m = ROCArray(ones(Float32, 2048, 32))
            v = vec(m)
            sum(v)
            sum(m; dims=1)
            sum(m; dims=2)
            cumsum(v)
            any(iszero, v)
            all(iszero, v)
            sum(ROCArray(ones(Int, 8192)))
            AMDGPU.synchronize()
        end
    end
end

end


end   # module AMDGPUExt
