module AcceleratedKernelsCUDAExt


import AcceleratedKernels as AK
import CUDA
using CUDA: @device_override


# On CUDA the NVPTX backend does not select scoped atomic fences, so `UnsafeAtomics.fence`
# with acquire/release ordering (the generic `AK._decoupled_fence`) either fails to lower or
# is not device-scope coherent. Provide the decoupled-lookback device fence with the native
# `membar.gl` threadfence instead.
@device_override @inline AK._decoupled_fence() = CUDA.threadfence()


end   # module AcceleratedKernelsCUDAExt
