### Accumulate / Prefix Sum / Scan


```@docs
AcceleratedKernels.accumulate!
AcceleratedKernels.accumulate
```

[`Auto()`](@ref AcceleratedKernels.Auto), the default `alg`, uses
[`CPUThreads.Partitioned`](@ref AcceleratedKernels.CPUThreads.Partitioned) on the host; on GPUs,
[`ScanPrefixes`](@ref AcceleratedKernels.ScanPrefixes) for whole arrays and
[`SliceScan`](@ref AcceleratedKernels.SliceScan) along `dims`, with the device's settings. Pass an
algorithm to choose it and its settings yourself:

```@docs
AcceleratedKernels.ScanAlgorithm
AcceleratedKernels.ScanPrefixes
AcceleratedKernels.DecoupledLookback
AcceleratedKernels.SliceScan
```

```julia
v = CuArray(rand(Int32(1):Int32(100), 1_000_000))
AK.accumulate!(+, v; init=Int32(0), alg=AK.ScanPrefixes(block_size=512))  # 512 threads, 8 items per thread
AK.accumulate!(+, v; init=Int32(0), alg=AK.DecoupledLookback())            # CUDA and AMDGPU only
m = CuArray(rand(Int32(1):Int32(100), 100, 10_000))
AK.accumulate(+, m; init=Int32(0), dims=2, alg=AK.SliceScan(block_size=128))
```
