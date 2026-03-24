using BenchmarkTools
using CUDA

import AcceleratedKernels as AK


const N = 100_000_000
const GPU_BLOCK_SIZE = 256

const RNG_SPLITMIX = AK.CounterRNG(0x12345678; alg=AK.SplitMix64())
const RNG_PHILOX = AK.CounterRNG(0x12345678; alg=AK.Philox())
const RNG_THREEFRY = AK.CounterRNG(0x12345678; alg=AK.Threefry())

x_cuda = CuArray{Float32}(undef, N)
x_splitmix = CuArray{Float32}(undef, N)
x_philox = CuArray{Float32}(undef, N)
x_threefry = CuArray{Float32}(undef, N)
x_cpu = Vector{Float32}(undef, N)


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


# Julia base rand() gives [0, 1) and so does EVERYTHING ELSE EVER! but CuRAND gives (0, 1] ...
is_unit_interval(v) = all(x -> 0.0f0 <= x <= 1.0f0, v)

# warmup compile
run_cuda_rand!(x_cuda)
# run_ak_rand_gpu!(RNG_SPLITMIX, x_splitmix)
run_ak_rand_gpu!(RNG_PHILOX, x_philox)
run_ak_rand_gpu!(RNG_THREEFRY, x_threefry)
run_ak_rand_cpu!(RNG_SPLITMIX, x_cpu)

@assert is_unit_interval(Array(x_cuda))
# @assert is_unit_interval(Array(x_splitmix))
@assert is_unit_interval(Array(x_philox))
@assert is_unit_interval(Array(x_threefry))
@assert is_unit_interval(x_cpu)

println("N = ", N)
println("CPU threads: ", Threads.nthreads())

println("\nCUDA.rand! benchmark (CuArray{Float32}, in-place)")
display(@benchmark run_cuda_rand!($x_cuda))

# println("\nAK.rand! SplitMix64 benchmark (GPU, CuArray{Float32})")
# display(@benchmark run_ak_rand_gpu!($RNG_SPLITMIX, $x_splitmix))

println("\nAK.rand! Philox benchmark (GPU, CuArray{Float32})")
display(@benchmark run_ak_rand_gpu!($RNG_PHILOX, $x_philox))

println("\nAK.rand! Threefry benchmark (GPU, CuArray{Float32})")
display(@benchmark run_ak_rand_gpu!($RNG_THREEFRY, $x_threefry))

println("\nAK.rand! benchmark (CPU, Vector{Float32}, SplitMix64)")
display(@benchmark run_ak_rand_cpu!($RNG_SPLITMIX, $x_cpu))

