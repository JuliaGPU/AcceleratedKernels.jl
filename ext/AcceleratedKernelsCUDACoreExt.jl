module AcceleratedKernelsCUDACoreExt

using CUDACore
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan.
CUDACore.@device_override AK._decoupled_fence() = CUDACore.threadfence()

end
