using BenchmarkTools
using CUDA

import AcceleratedKernels as AK


const N = 100_000_000
const GPU_BLOCK_SIZE = 256


const RNG_PHILOX = AK.CounterRNG(0x12345678; alg=AK.Philox())


TestType = Float32
x_cuda = CuArray{TestType}(undef, N)
x_philox = CuArray{TestType}(undef, N)
x_cpu = Vector{TestType}(undef, N)


function run_cuda_rand!(x)
    CUDA.rand!(x)
    CUDA.synchronize()
    return x
end


function run_ak_rand_gpu!(rng, x)
    AK.rand!(rng, x; block_size=GPU_BLOCK_SIZE)
    AK.synchronize(AK.get_backend(x))
    return x
end


function run_ak_rand_cpu!(rng, x)
    AK.rand!(rng, x)
    return x
end


# warmup
run_cuda_rand!(x_cuda)
run_ak_rand_gpu!(RNG_PHILOX, x_philox)
run_ak_rand_cpu!(RNG_PHILOX, x_cpu)


println("N = ", N)
println("CPU threads: ", Threads.nthreads())

println("\nCUDA.rand! benchmark (CuArray{$TestType}, in-place)")
display(@benchmark run_cuda_rand!($x_cuda))

println("\nAK.rand! Philox benchmark (GPU, CuArray{$TestType})")
display(@benchmark run_ak_rand_gpu!($RNG_PHILOX, $x_philox))

println("\nAK.rand! benchmark (CPU, Vector{$TestType}, Philox)")
display(@benchmark run_ak_rand_cpu!($RNG_PHILOX, $x_cpu))

