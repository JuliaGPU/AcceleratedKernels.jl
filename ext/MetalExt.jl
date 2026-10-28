module MetalExt

using Metal
import AcceleratedKernels as AK

# Scan tiles fit half of the 32 KiB of threadgroup memory. Metal's shader validation doubles every
# kernel's threadgroup memory, so larger tiles can't run under it, and they aren't faster: on an
# M-series GPU, scanning 10^7 Int64 or ComplexF32 elements takes 8% to 10% less time with these
# tiles than with 32 KiB ones.
# Partial results flag whether they are set with a `Bool`: with the wider flag, scans of 10^6 to
# 10^8 Float32s without a neutral element along `dims` take up to 1.15x as long on an M1. (Apple
# GPUs allocate registers in 16-bit halves, so a `UInt32` flag takes one half more than a `Bool`, which
# pushes `slice_tiles_kernel!` from 67 to 70 registers and over an occupancy step.)
AK.scan_tuning(::MetalBackend, ::Type) =
    AK.ScanTuning(local_mem_bytes=16 * 1024, wide_optional_flag=false)

# `ConcurrentWrite` launches a thread per element, which Metal's grid cannot hold for 2^32
# elements or more (#156)
AK.predicate_tuning(::MetalBackend, ::Type) = AK.PredicateTuning(prefer_concurrent_write=false)

# Reductions in blocks of 64 threads: on an M1, summing a 512×1000 matrix along either dimension
# takes 15% less time than with 256 threads, as does `findfirst` over 512k elements, while
# reductions of 16M elements take at most 4% more. Groups of threads read runs of up to 2048
# Float32s, which takes about 20% less time than a block per output on an M1 (and ties at 4096).
AK.reduce_tuning(::MetalBackend, ::Type) =
    AK.ReduceTuning(block_size=64, lanes_max_bytes=256, wide_optional_flag=false)

# Sorting random keys on an M1. `BitonicSort` sorts whole arrays of up to 512 elements 1.03-1.4x
# faster than `MergeSort` (every sort takes at least about 0.2 ms), and slices of up to 1024
# elements along `dims` 1.0-13x faster in arrays of 2^16 elements or more (up to 10% slower in
# arrays of 2^12). `RadixSort` sorts whole arrays of 2^17 32-bit elements 1.4x faster and of 2^18
# 64-bit elements 1.1x faster (1.9x at 2^23). Along `dims`, it sorts 32-bit slices of 2^15
# elements in arrays of 2^17 or more 1.1-3.3x faster; 64-bit slices need 10^5 to 10^6 elements.
AK.sort_tuning(::MetalBackend, ::Type{T}) where {T} =
    sizeof(T) <= 4 ?
    AK.SortTuning(bitonic_max_len=512, radix_min_len=1 << 17, bitonic_max_slice_len=1024,
                  radix_min_slice_len=1 << 15) :
    AK.SortTuning(bitonic_max_len=512, radix_min_len=1 << 18, bitonic_max_slice_len=1024)

end
