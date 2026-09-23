### Reductions

Apply a custom binary operator reduction on all elements in an iterable; can be used to compute minima, sums, counts, etc.
- **Other names**: `Kokkos:parallel_reduce`, `fold`, `aggregate`.

---

```@docs
AcceleratedKernels.reduce
```

The operator must be associative and commutative: like every GPU reduction, AcceleratedKernels
combines elements in an order that depends on the algorithm and its settings, not in element
order.

[`Auto()`](@ref AcceleratedKernels.Auto), the default `alg`, uses
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned) on the host and
[`BlockReduce`](@ref AcceleratedKernels.BlockReduce) on GPUs, with the device's settings. Pass an
algorithm to choose its settings yourself:

```@docs
AcceleratedKernels.ReduceAlgorithm
AcceleratedKernels.BlockReduce
AcceleratedKernels.CPUThreads.Partitioned
```

```julia
m = ROCArray(rand(Float32, 1000, 1000))
AK.reduce(+, m; init=0f0, alg=AK.BlockReduce(block_size=512))            # 512 threads per block
AK.reduce(+, m; init=0f0, alg=AK.BlockReduce(switch_below=1024))         # finish on the host below 1024 values
AK.reduce(+, Array(m); init=0f0, alg=AK.CPUThreads.Partitioned(max_tasks=4))
```
