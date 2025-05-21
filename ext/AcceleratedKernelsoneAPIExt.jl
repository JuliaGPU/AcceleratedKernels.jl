module AcceleratedKernelsoneAPIExt


using oneAPI
import AcceleratedKernels as AK


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


end   # module AcceleratedKernelsoneAPIExt