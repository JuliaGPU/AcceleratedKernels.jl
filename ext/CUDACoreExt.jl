module CUDACoreExt

using CUDACore
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan.
CUDACore.@device_override AK._decoupled_fence() = CUDACore.threadfence()

# The fence above orders device-scope memory, and blocks make forward progress while others
# wait on them: DecoupledLookback is correct here.
AK._supports_lookback(::CUDABackend) = true

# Groups of threads read runs of up to 4096 Float32s: on an RTX 5080, summing a 4096×4096 matrix
# along `dims=1` takes 17% less time than with a block per output.
AK.reduce_tuning(::CUDABackend, ::Type) = AK.ReduceTuning(segments_lane_bytes=512)

end
