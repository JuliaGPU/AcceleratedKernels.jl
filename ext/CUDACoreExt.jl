module CUDACoreExt

using CUDACore
import AcceleratedKernels as AK

# Groups of threads read runs of up to 4096 Float32s: on an RTX 5080, summing a 4096×4096 matrix
# along `dims=1` takes 17% less time than with a block per output.
AK.reduce_tuning(::CUDABackend, ::Type) = AK.ReduceTuning(lanes_max_bytes=512)

# Sorting random keys on an RTX 5080 and an RTX 6000 Ada. `BitonicSort` sorts whole arrays of up to
# 512 elements 1.15-1.8x faster than `MergeSort`, and many slices of up to 256 elements along
# `dims` 1.1-66x faster; on the Ada, slices of 362 are 1-10% slower, and with fewer than 2^16
# elements in all, slices of 64 to 181 take up to 41% (a few µs) longer. `RadixSort` sorts whole
# arrays of 2^22 32-bit elements 2-2.2x faster on the 5080 and 1.2-1.6x on the Ada, and of 2^23
# 64-bit elements 1.5-3.2x and 1.0-2.3x faster (Float64 most, which slows `MergeSort` down on
# these GPUs). Along `dims`, it sorts 32-bit slices of 2^19 elements in arrays of 2^22 1.0-3.7x
# faster (Int32 along `dims=1` on the Ada least), but never wins for 64-bit integers on the Ada.
AK.sort_tuning(::CUDABackend, ::Type{T}) where {T} =
    sizeof(T) <= 4 ?
    AK.SortTuning(bitonic_max_len=512, radix_min_len=1 << 22, bitonic_max_slice_len=256,
                  radix_min_slice_len=1 << 19) :
    AK.SortTuning(bitonic_max_len=512, radix_min_len=1 << 23, bitonic_max_slice_len=256)

end
