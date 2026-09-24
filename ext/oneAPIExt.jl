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


# On oneAPI, use the MapReduce algorithm by default as on some Intel GPUs ConcurrentWrite hangs
# the device.
function AK.any(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # Algorithm choice
    alg::AK.PredicatesAlgorithm=AK.MapReduce(),
    kwargs...
)
    AK._any_impl(
        pred, v, backend;
        alg,
        kwargs...
    )
end


function AK.all(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # Algorithm choice
    alg::AK.PredicatesAlgorithm=AK.MapReduce(),
    kwargs...
)
    AK._all_impl(
        pred, v, backend;
        alg,
        kwargs...
    )
end


end   # module oneAPIExt
