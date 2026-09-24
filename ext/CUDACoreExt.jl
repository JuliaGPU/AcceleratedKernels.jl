module CUDACoreExt

using CUDACore
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan.
CUDACore.@device_override AK._decoupled_fence() = CUDACore.threadfence()

# The fence above orders device-scope memory, and blocks make forward progress while others
# wait on them: DecoupledLookback is correct here.
AK._supports_lookback(::CUDABackend) = true

end
