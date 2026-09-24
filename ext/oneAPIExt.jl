module oneAPIExt


using oneAPI
using oneAPI: method_table          # used by oneAPI.@device_override
import AcceleratedKernels as AK


# Device-scope SPIR-V fence for the DecoupledLookback scan.
const SPIRV = oneAPI.SPIRVIntrinsics
oneAPI.@device_override AK._decoupled_fence() =
    SPIRV.atomic_work_item_fence(SPIRV.GLOBAL_MEM_FENCE, SPIRV.memory_order_seq_cst, SPIRV.memory_scope_device)


# WORKAROUND(oneAPI): scans use 64-thread blocks, as oneAPI.jl did (1e70e3b), because they give
# wrong results on Intel GPUs with blocks of 128 threads or more; the root cause is not known yet.
# Once fixed, drop this method.
AK.scan_tuning(::oneAPIBackend, ::Type) = AK.ScanTuning(block_size=64)


# WORKAROUND(oneAPI): some Intel GPUs hang when many threads write one global location, and the
# affected devices are not known, so `any`/`all` use `ViaReduce` on every oneAPI device (as
# oneAPI.jl did). Once the devices are known, declare only those.
AK._supports_concurrent_write(::oneAPIBackend) = false


end   # module oneAPIExt
