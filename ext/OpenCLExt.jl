module OpenCLExt

using OpenCL
using OpenCL: method_table          # used by OpenCL.@device_override
import AcceleratedKernels as AK

# Device-scope SPIR-V fence for the DecoupledLookback scan (also the POCL path).
const SPIRV = OpenCL.SPIRVIntrinsics
OpenCL.@device_override AK._decoupled_fence() =
    SPIRV.atomic_work_item_fence(SPIRV.GLOBAL_MEM_FENCE, SPIRV.memory_order_seq_cst, SPIRV.memory_scope_device)

# CPU devices (e.g. PoCL) reduce short runs faster with a thread per output, and long ones faster
# with groups of threads: with PoCL on a Ryzen 9950X, summing an 8×10^6 matrix along `dims=1`
# takes half the time, and a 4096×4096 one along `dims=1` also half the time.
function AK.reduce_tuning(::OpenCLBackend, ::Type)
    cl.device().device_type === :cpu || return AK.ReduceTuning()
    return AK.ReduceTuning(segments_short_runs=false, segments_lane_bytes=512)
end

# CPU devices (e.g. PoCL) implement sub-group operations with barriers of the whole block, and
# keep a scanned tile in their caches for adding its prefix
function AK.scan_tuning(::OpenCLBackend, ::Type)
    cl.device().device_type === :cpu || return AK.ScanTuning()
    return AK.ScanTuning(reduce_then_scan=false, subgroups=false)
end

end
