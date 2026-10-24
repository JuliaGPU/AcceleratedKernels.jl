module OpenCLExt

using OpenCL
import AcceleratedKernels as AK

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
