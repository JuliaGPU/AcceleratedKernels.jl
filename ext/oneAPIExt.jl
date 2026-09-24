module oneAPIExt


using oneAPI
using oneAPI: method_table          # used by oneAPI.@device_override
import AcceleratedKernels as AK


# Device-scope SPIR-V fence for the DecoupledLookback scan.
const SPIRV = oneAPI.SPIRVIntrinsics
oneAPI.@device_override AK._decoupled_fence() =
    SPIRV.atomic_work_item_fence(SPIRV.GLOBAL_MEM_FENCE, SPIRV.memory_order_seq_cst, SPIRV.memory_scope_device)


# Some Intel GPUs (reportedly Intel UHD Graphics) hang when many threads write one global location,
# as `ConcurrentWrite` does. An Iris Xe does not, but the affected devices are not known, so `Auto`
# keeps using `ViaReduce` for `any`/`all` on oneAPI; an explicit `ConcurrentWrite` is allowed.
AK.predicate_tuning(::oneAPIBackend, ::Type) = AK.PredicateTuning(prefer_concurrent_write=false)


end   # module oneAPIExt
