### Predicates

Apply a predicate to check if all / any elements in a collection return true. Could be implemented as a reduction, but is better optimised with stopping the search once a false / true is found. GPU libraries rarely provide these on their own; they are usually written as reductions.


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
instead. Metal does too, as `ConcurrentWrite` launches a thread per element, more than Metal's grid
holds for arrays of 2^32 elements or more. An explicit `alg=ConcurrentWrite()` runs on every GPU.

```@docs
AcceleratedKernels.PredicateAlgorithm
AcceleratedKernels.ConcurrentWrite
AcceleratedKernels.ViaReduce
```
