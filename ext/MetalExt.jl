module MetalExt

using Metal
import AcceleratedKernels as AK

# Device-scope fence for the DecoupledLookback scan (Metal 3.2+).
Metal.@device_override AK._decoupled_fence() =
    Metal.atomic_thread_fence(Metal.MemoryFlagDevice, Metal.memory_order_seq_cst, Metal.thread_scope_device)

end
