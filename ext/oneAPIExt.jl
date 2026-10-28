module oneAPIExt


using oneAPI
import AcceleratedKernels as AK
using PrecompileTools: @setup_workload, @compile_workload


# Some Intel GPUs (reportedly Intel UHD Graphics) hang when many threads write one global location,
# as `ConcurrentWrite` does. An Iris Xe does not, but the affected devices are not known, so `Auto`
# keeps using `ViaReduce` for `any`/`all` on oneAPI; an explicit `ConcurrentWrite` is allowed.
AK.predicate_tuning(::oneAPIBackend, ::Type) = AK.PredicateTuning(prefer_concurrent_write=false)

# Launches cost tens of microseconds more than on other GPUs (Iris Xe: `sum(A; dims=2)` of a
# 64×4096 matrix takes 220 µs in one launch and 280 µs in the column reduction's two; from about
# 4M elements the column reduction is faster), so the column reduction starts at larger inputs.
AK.reduce_tuning(::oneAPIBackend, ::Type) = AK.ReduceTuning(columns_split_min_elems=1 << 22)

# Sorting random keys on an Arc A770, where every sort takes at least about 0.1 ms. `BitonicSort`
# sorts whole arrays of 23 to 128 elements 1.0-1.4x faster than `MergeSort` (16 elements up to
# 23% slower), and slices of up to 128 elements along `dims` 1.0-55x faster in arrays of 2^16
# elements or more (up to 69% slower in arrays of 2^12); longer slices lose by up to 5%.
# `RadixSort` sorts whole arrays of 2^21 32-bit elements 1.6-1.8x faster, and 32-bit slices of
# 2^15 elements in arrays of 2^21 or more 1.1-2.8x faster. For 64-bit elements it ranges from
# 0.4x to 1.5x up to 2^25 elements.
AK.sort_tuning(::oneAPIBackend, ::Type{T}) where {T} =
    sizeof(T) <= 4 ?
    AK.SortTuning(bitonic_max_len=128, radix_min_len=1 << 21, bitonic_max_slice_len=128,
                  radix_min_slice_len=1 << 15) :
    AK.SortTuning(bitonic_max_len=128, bitonic_max_slice_len=128)


# oneAPI's own workload can't cache these operations: this extension isn't loaded while oneAPI
# precompiles, so the kernels it would compile use the default tunings, not the ones above. The
# sizes are large enough for the kernels that arrays of millions of elements use. Before Julia
# 1.12, compiling GPU code while precompiling can leak device code into the image.
@static if VERSION >= v"1.12-"
@setup_workload begin
    # (`can_precompile` checks that the GPU can be used while precompiling)
    if isdefined(oneAPI, :can_precompile) && oneAPI.can_precompile()
        @compile_workload begin
            m = oneArray(ones(Float32, 4096, 64))
            v = vec(m)
            sum(v)
            sum(m; dims=1)
            sum(m; dims=2)
            cumsum(v)
            any(iszero, v)
            all(iszero, v)
            sum(oneArray(ones(Int, 4096)))
            oneAPI.synchronize()
        end
    end
end
end


end   # module oneAPIExt
