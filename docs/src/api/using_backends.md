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

All publicly-exposed functions have CPU implementations with unified parameter interfaces:

```julia
import AcceleratedKernels as AK
v = Vector(-1000:1000)                          # Normal CPU array
AK.reduce(+, v, max_tasks=Threads.nthreads())
```

By default all algorithms use the number of threads Julia was started with.
