### Predicates

Apply a predicate to check if all / any elements in a collection return true. Could be implemented as a reduction, but is better optimised with stopping the search once a false / true is found.
- **Other names**: not often implemented standalone on GPUs, typically included as part of a reduction.


```@docs
AcceleratedKernels.any
AcceleratedKernels.all
```

[`Auto()`](@ref AcceleratedKernels.Auto) uses
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned) on the host and
[`ConcurrentWrite`](@ref AcceleratedKernels.ConcurrentWrite) on GPUs, except where concurrent
writes to one location are unsafe: some older platforms (old Intel Graphics) hang when many
threads write the same memory location, even with the same value, which is well-defined on others
(CUDA F4.2: "If a non-atomic instruction executed by a warp writes to the same location in global
memory for more than one of the threads of the warp, only one thread performs a write and which
thread does it is undefined."). The oneAPI extension declares concurrent writes unsafe on every
oneAPI device, where `Auto` uses the `mapreduce`-based
[`ViaReduce`](@ref AcceleratedKernels.ViaReduce) instead; other backends, including OpenCL on Intel
GPUs, use `ConcurrentWrite` unless an extension says otherwise.

```@docs
AcceleratedKernels.PredicateAlgorithm
AcceleratedKernels.ConcurrentWrite
AcceleratedKernels.ViaReduce
```
