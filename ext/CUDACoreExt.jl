module CUDACoreExt

using CUDACore
import AcceleratedKernels as AK

# Groups of threads read runs of up to 4096 Float32s: on an RTX 5080, summing a 4096×4096 matrix
# along `dims=1` takes 17% less time than with a block per output.
AK.reduce_tuning(::CUDABackend, ::Type) = AK.ReduceTuning(segments_lane_bytes=512)

end
