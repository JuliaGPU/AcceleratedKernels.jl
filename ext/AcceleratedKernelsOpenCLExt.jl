module AcceleratedKernelsOpenCLExt

using OpenCL
using OpenCL: method_table          # used by OpenCL.@device_override
import AcceleratedKernels as AK

# Device-scope SPIR-V fence for the DecoupledLookback scan (also the POCL path).
const SPIRV = OpenCL.SPIRVIntrinsics
OpenCL.@device_override AK._decoupled_fence() =
    SPIRV.atomic_work_item_fence(SPIRV.GLOBAL_MEM_FENCE, SPIRV.memory_order_seq_cst, SPIRV.memory_scope_device)

end
