using BenchmarkTools
using CUDA

import AcceleratedKernels as AK


const N = 100_000_000
const GPU_BLOCK_SIZE = 256

const RNG_PHILOX = AK.CounterRNG(0x12345678; alg=AK.Philox(), offset=0x0)

TestType = Float32

x_cuda = CuArray{TestType}(undef, N)
x_philox = CuArray{TestType}(undef, N)
x_cpu = Vector{TestType}(undef, N)


function run_cuda_randn!(x)
    CUDA.randn!(x)
    CUDA.synchronize()
    return x
end


function run_ak_randn_gpu!(rng, x)
    AK.randn!(rng, x; block_size=GPU_BLOCK_SIZE)
    AK.synchronize(AK.get_backend(x))
    return x
end


function run_ak_randn_cpu!(rng, x)
    AK.randn!(rng, x)
    return x
end

# warmup compile
run_cuda_randn!(x_cuda)
run_ak_randn_gpu!(RNG_PHILOX, x_philox)

println("N = ", N)
println("CPU threads: ", Threads.nthreads())

println("\nCUDA.randn! benchmark (CuArray{$TestType}, in-place)")
display(@benchmark run_cuda_randn!($x_cuda))

println("\nAK.randn! Philox benchmark (GPU, CuArray{$TestType})")
display(@benchmark run_ak_randn_gpu!($RNG_PHILOX, $x_philox))


