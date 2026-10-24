module MetalExt

using Metal
import AcceleratedKernels as AK

# Scan tiles fit half of the 32 KiB of threadgroup memory. Metal's shader validation doubles every
# kernel's threadgroup memory, so larger tiles can't run under it, and they aren't faster: on an
# M-series GPU, scanning 10^7 Int64 or ComplexF32 elements takes 8% to 10% less time with these
# tiles than with 32 KiB ones.
AK.scan_tuning(::MetalBackend, ::Type) = AK.ScanTuning(local_mem_bytes=16 * 1024)

# `ConcurrentWrite` launches a thread per element, which Metal's grid cannot hold for 2^32
# elements or more (#156)
AK.predicate_tuning(::MetalBackend, ::Type) = AK.PredicateTuning(prefer_concurrent_write=false)

# Reductions in blocks of 64 threads: on an M1, summing a 512×1000 matrix along either dimension
# takes 15% less time than with 256 threads, as does `findfirst` over 512k elements, while
# reductions of 16M elements take at most 4% more. Groups of threads read runs of up to 2048
# Float32s, which takes about 20% less time than a block per output on an M1 (and ties at 4096).
AK.reduce_tuning(::MetalBackend, ::Type) = AK.ReduceTuning(block_size=64, segments_lane_bytes=256)

end
