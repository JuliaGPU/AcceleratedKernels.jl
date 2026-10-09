module MetalExt

using Metal
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan (Metal 3.2+).
Metal.@device_override AK._decoupled_fence() =
    Metal.atomic_thread_fence(Metal.MemoryFlagDevice, Metal.memory_order_seq_cst, Metal.thread_scope_device)

# Scan tiles fit half of the 32 KiB of threadgroup memory. Metal's shader validation doubles every
# kernel's threadgroup memory, so larger tiles can't run under it, and they aren't faster: on an
# M-series GPU, scanning 10^7 Int64 or ComplexF32 elements takes 8% to 10% less time with these
# tiles than with 32 KiB ones.
AK.scan_tuning(::MetalBackend, ::Type) = AK.ScanTuning(local_mem_bytes=16 * 1024)

end
