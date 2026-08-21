module AcceleratedKernelsAMDGPUExt

using AMDGPU
import UnsafeAtomics
import AcceleratedKernels as AK

# Device-scope (agent) fence for the DecoupledLookback scan.
AMDGPU.Device.@device_override AK._decoupled_fence() =
    UnsafeAtomics.fence(UnsafeAtomics.seq_cst, AMDGPU.syncscope_agent)

end
