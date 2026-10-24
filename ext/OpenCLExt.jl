module OpenCLExt

using OpenCL
import AcceleratedKernels as AK

# CPU devices (e.g. PoCL) get the tunings of the host backend's kernels
AK.reduce_tuning(::OpenCLBackend, ::Type) =
    cl.device().device_type === :cpu ? AK.cpu_reduce_tuning() : AK.ReduceTuning()
AK.scan_tuning(::OpenCLBackend, ::Type) =
    cl.device().device_type === :cpu ? AK.cpu_scan_tuning() : AK.ScanTuning()

end
