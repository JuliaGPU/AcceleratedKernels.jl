module MetalExt

using Metal
import AcceleratedKernels as AK
using PrecompileTools: @setup_workload, @compile_workload

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
AK.reduce_tuning(::MetalBackend, ::Type) = AK.ReduceTuning(block_size=64, lanes_max_bytes=256)

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


# Metal's own workload can't cache AcceleratedKernels' operations as they run with the tunings
# above: this extension isn't loaded while Metal precompiles, so the kernels compiled there use
# the default tunings. Before Julia 1.11, GPU code cached in a package image shadows the host
# versions of the methods it shares with host code (see Metal's workload).
@static if VERSION >= v"1.11"
@setup_workload begin
    # only when Metal's own workload launches kernels too (older versions of Metal can't tell,
    # and implement some of these operations differently, so skip the workload there)
    can_launch = isdefined(Metal, :can_precompile_launches) && Metal.can_precompile_launches()
    @compile_workload if can_launch
        # the vector spans several blocks, so that whole-array reductions take both passes, and
        # the matrix is large enough for the kernels that reduce large matrices along dims
        let m = MtlArray(ones(Float32, 256, 32)), v = vec(m)
            sum(v)
            sum(m; dims=1)
            sum(m; dims=2)
            cumsum(v)
            any(iszero, v)
            all(iszero, v)
        end
        sum(MtlArray(ones(Int, 1024)))
        synchronize()
    end
end
end

end
