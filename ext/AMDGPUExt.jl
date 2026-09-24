module AMDGPUExt

using AMDGPU
import UnsafeAtomics
import AcceleratedKernels as AK

# Device-scope (agent) fence for the DecoupledLookback scan.
AMDGPU.Device.@device_override AK._decoupled_fence() =
    UnsafeAtomics.fence(UnsafeAtomics.seq_cst, AMDGPU.syncscope_agent)

# The fence above orders device-scope memory, and blocks make forward progress while others
# wait on them: DecoupledLookback is correct here.
AK._supports_lookback(::ROCBackend) = true

end
