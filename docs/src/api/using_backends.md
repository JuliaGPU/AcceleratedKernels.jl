### Using Different Backends

For any of the examples here, simply use a different GPU array and AcceleratedKernels.jl will pick the right backend:
```julia
# Intel Graphics
using oneAPI
v = oneArray{Int32}(undef, 100_000)             # Empty array

# AMD ROCm
using AMDGPU
v = ROCArray{Float64}(1:100_000)                # A range converted to Float64

# Apple Metal
using Metal
v = MtlArray(rand(Float32, 100_000))            # Transfer from host to device

# NVidia CUDA
using CUDA
v = CuArray{UInt32}(0:5:100_000)                # Range with explicit step size

# Transfer GPU array back
v_host = Array(v)
```

All publicly-exposed functions also run on host arrays, with the same interface:

```julia
import AcceleratedKernels as AK
v = Vector(-1000:1000)                          # Normal CPU array
AK.reduce(+, v; init=0)
AK.reduce(+, v; init=0, alg=AK.CPUThreads.Partitioned(max_tasks=4))   # at most 4 tasks
```

On the host, operations run on Julia threads and by default use as many tasks as Julia has
threads. See [Algorithms and backends](@ref) for the algorithms and their settings.
