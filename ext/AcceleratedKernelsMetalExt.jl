module AcceleratedKernelsMetalExt


using Metal
import AcceleratedKernels as AK


# On Metal use the ScanPrefixes accumulation algorithm by default as the DecoupledLookback algorithm
# cannot be supported due to Metal's weaker memory consistency guarantees.
function AK.accumulate!(
    op, v::AbstractArray, backend::MetalBackend;
    init,
    # Algorithm choice is the only differing default
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),
    kwargs...
)
    AK._accumulate_impl!(
        op, v, backend;
        init, alg,
        kwargs...
    )
end


# Two-argument version for compatibility with Base.accumulate!
function AK.accumulate!(
    op, dst::AbstractArray, src::AbstractArray, backend::MetalBackend;
    init,
    # Algorithm choice is the only differing default
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),
    kwargs...
)
    copyto!(dst, src)
    AK._accumulate_impl!(
        op, dst, backend;
        init, alg,
        kwargs...
    )
end

end   # module AcceleratedKernelsMetalExt