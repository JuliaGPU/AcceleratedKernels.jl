module OpenCLExt

using OpenCL
import AcceleratedKernels as AK

# CPU devices (e.g. PoCL) get the tunings of the host backend's kernels
AK.reduce_tuning(::OpenCLBackend, ::Type) =
    cl.device().device_type === :cpu ? AK.cpu_reduce_tuning() : AK.ReduceTuning()
AK.scan_tuning(::OpenCLBackend, ::Type) =
    cl.device().device_type === :cpu ? AK.cpu_scan_tuning() : AK.ScanTuning()

# Sorting random keys with PoCL on a Ryzen 9 9950X. `BitonicSort` sorts whole arrays of up to 128
# elements 1.4-1.9x faster than `MergeSort`, and slices of up to 1024 elements along `dims`
# 1.4-40x faster in arrays of 2^16 elements or more (up to 39% slower in arrays of 2^12).
# `RadixSort` sorts whole arrays of 2^17 32-bit elements 1.9-2.2x faster and of 2^19 64-bit
# elements 1.7-2x faster; along `dims`, it sorts 32-bit slices of 512 elements and 64-bit slices
# of 2^18 in arrays of those sizes or larger 1.05-7.7x faster.
# GPU devices have not been measured, so they keep `MergeSort`.
function AK.sort_tuning(::OpenCLBackend, ::Type{T}) where {T}
    cl.device().device_type === :cpu || return AK.SortTuning()
    sizeof(T) <= 4 ?
    AK.SortTuning(bitonic_max_len=128, radix_min_len=1 << 17, bitonic_max_slice_len=1024,
                  radix_min_slice_len=512) :
    AK.SortTuning(bitonic_max_len=128, radix_min_len=1 << 19, bitonic_max_slice_len=1024,
                  radix_min_slice_len=1 << 18)
end

end
