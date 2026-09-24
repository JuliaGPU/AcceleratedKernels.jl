### Predicates

Apply a predicate to check if all / any elements in a collection return true. Could be implemented as a reduction, but is better optimised with stopping the search once a false / true is found.
- **Other names**: not often implemented standalone on GPUs, typically included as part of a reduction.


```@docs
AcceleratedKernels.any
AcceleratedKernels.all
```

[`Auto()`](@ref AcceleratedKernels.Auto) uses
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned) on the host and
[`ConcurrentWrite`](@ref AcceleratedKernels.ConcurrentWrite) on GPUs, in which many threads write
the same value to one memory location. That is well-defined (CUDA F4.2: "If a non-atomic
instruction executed by a warp writes to the same location in global memory for more than one of
the threads of the warp, only one thread performs a write and which thread does it is
undefined."), but some older platforms (Intel UHD Graphics) have been reported to hang on it, so
on oneAPI `Auto` uses the `mapreduce`-based [`ViaReduce`](@ref AcceleratedKernels.ViaReduce)
instead. An explicit `alg=ConcurrentWrite()` runs on every GPU.

```@docs
AcceleratedKernels.PredicateAlgorithm
AcceleratedKernels.ConcurrentWrite
AcceleratedKernels.ViaReduce
```
