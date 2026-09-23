### General Looping: `foreachindex` / `foraxes`

General workhorses for converting normal Julia `for` loops into GPU code, for example:

```julia
# Copy kernel testing throughput
function cpu_copy!(dst, src)
    for i in eachindex(src)
        dst[i] = src[i]
    end
end
```

Would be written for GPU as:

```julia
import AcceleratedKernels as AK

function gpu_copy!(dst, src)
    AK.foreachindex(src) do i
        dst[i] = src[i]
    end
end
```

Yes, simply change `for i in eachindex(itr)` into `AK.foreachindex(itr) do i` to run it on GPUs / multithreaded - magic! (or just amazing language design)

This is a parallelised for-loop over the indices of an iterable; converts normal Julia code to GPU kernels running one thread per index. On CPUs it executes static index ranges on `max_tasks` threads, with user-defined `min_elems` to be processed by each thread; if only a single thread ends up being needed, the loop is inlined and executed without spawning threads.
- **Other names**: `Kokkos::parallel_for`, `RAJA::forall`, `thrust::transform`.


Example:
```julia
import AcceleratedKernels as AK

function f(a, b)
    # Don't use global arrays inside a `foreachindex`; types must be known
    @assert length(a) == length(b)
    AK.foreachindex(a) do i
        # Note that we don't have to explicitly pass b into the lambda
        if b[i] > 0.5
            a[i] = 1
        else
            a[i] = 0
        end

        # Showing arbitrary if conditions; can also be written as:
        # @inbounds a[i] = b[i] > 0.5 ? 1 : 0
    end
end

# Use any backend, e.g. CUDA, ROCm, oneAPI, Metal, or CPU
using oneAPI
v1 = oneArray{Float32}(undef, 100_000)
v2 = oneArray(rand(Float32, 100_000))
f(v1, v2)
```

The loop runs on the backend of the iterable. A range does not determine one, so pass `backend` when a loop over a range accesses GPU arrays:
```julia
AK.foreachindex(1:n; backend=AK.get_backend(v)) do i
    v[i] = i
end
```
Host arrays and ranges run on Julia threads.

All GPU functions allow you to specify a block size - this is often a power of two (mostly 64, 128, 256, 512); the optimum depends on the algorithm, input data and hardware - you can try the different values and `@time` or `@benchmark` them:
```julia
@time AK.foreachindex(f, itr_gpu; block_size=512)
```

Similarly, for performance on the CPU the overhead of spawning threads should be masked by processing more elements per thread (but there is no reason here to launch more threads than `Threads.nthreads()`, the number of threads Julia was started with); the optimum depends on how expensive `f` is - again, benchmarking is your friend:
```julia
@time AK.foreachindex(f, itr_cpu; max_tasks=16, min_elems=1000)
```


```@docs
AcceleratedKernels.foreachindex
AcceleratedKernels.foraxes
```