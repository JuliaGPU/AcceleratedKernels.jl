### Reductions

Apply a custom binary operator reduction on all elements in an iterable; can be used to compute minima, sums, counts, etc.
- **Other names**: `Kokkos:parallel_reduce`, `fold`, `aggregate`.

---

```@docs
AcceleratedKernels.reduce
```

The operator must be associative and commutative: like every GPU reduction, AcceleratedKernels
combines elements in an order that depends on the algorithm and its settings, not in element
order. `init` is optional and applied exactly once; without it, an empty reduction is an error,
except for `sum`, `prod` and `count`, and operators without a known neutral element need none.
Results have the accumulator type, which `acctype` can set (see
[`mapreducedim!`](@ref AcceleratedKernels.mapreducedim!) for the contract, and
[Differences from Base](@ref) for how it differs from Base's).

```julia
AK.sum(CuArray(Int32[]))                # 0, an Int
AK.minimum(CuArray(Int32[]))            # an ArgumentError
AK.reduce((a, b) -> a + b, v)           # no neutral element needed
AK.reduce(+, v; init=10)                # init is added exactly once
AK.sum(CuArray(rand(Float32, 10^6)); acctype=Float64)   # summed in Float64
```

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
AK.reduce(+, m; alg=AK.BlockReduce(block_size=512))            # 512 threads per block
AK.reduce(+, m; alg=AK.BlockReduce(switch_below=1024))         # finish on the host below 1024 values
AK.reduce(+, Array(m); alg=AK.CPUThreads.Partitioned(max_tasks=4))
```
