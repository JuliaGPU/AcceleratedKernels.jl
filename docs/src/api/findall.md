### Find All / Stream Compaction

```@docs
AcceleratedKernels.findall
```

`findall` selects items: indices by default, or anything else of the input's length, such as
the values themselves.

```julia
v = CuArray(rand(Float32, 1000))
AK.findall(x -> x > 0.5f0, v)                  # indices
AK.findall(x -> x > 0.5f0, v; items=v)         # the selected values, as v[v .> 0.5f0]
```

[`Auto()`](@ref AcceleratedKernels.Auto) uses
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned) on the host and
[`ScanScatter`](@ref AcceleratedKernels.ScanScatter) on GPUs, with the device's settings.

```@docs
AcceleratedKernels.FindallAlgorithm
AcceleratedKernels.ScanScatter
```
