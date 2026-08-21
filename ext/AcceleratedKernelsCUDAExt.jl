module AcceleratedKernelsCUDAExt

using CUDA
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan.
CUDA.@device_override AK._decoupled_fence() = CUDA.threadfence()

end
