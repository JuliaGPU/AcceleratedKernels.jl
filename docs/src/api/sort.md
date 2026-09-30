###  `sort` and friends

Sorting with the interface and results of the Julia Base functions, on GPUs and on multithreaded CPUs:
- `sort!` (in-place), `sort` (out-of-place)
- `sortperm!`, `sortperm`
- `sort_by_key!`, which sorts keys and applies the same permutation to values
- **Other names**: `sort`, `sort_team`, `sort_team_by_key`, `stable_sort`, `sort_by_key` or variations in Kokkos, RAJA, Thrust, oneDPL and CUB.

Function signatures:
```@docs
AcceleratedKernels.sort!
AcceleratedKernels.sort
AcceleratedKernels.sortperm!
AcceleratedKernels.sortperm
AcceleratedKernels.sort_by_key!
```

Example:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32, 100_000))
AK.sort!(v)
```

Like Base, the sorts are stable by default: elements that compare equal keep their order.
[`Auto()`](@ref AcceleratedKernels.Auto), the default `alg`, picks an algorithm for the backend,
the device, the element type, the ordering and the length. Pass an algorithm to choose it and its
settings yourself; an algorithm that cannot run the call is an `ArgumentError`, never replaced:

| Algorithm | Stable | `dims` | `sortperm!` | `sort_by_key!` | Element types, orderings | Backends |
|---|---|---|---|---|---|---|
| [`MergeSort`](@ref AcceleratedKernels.MergeSort) | yes | yes | yes | yes | all | kernels |
| [`RadixSort`](@ref AcceleratedKernels.RadixSort) | yes | no | no | no | 32/64-bit integers and floats, default ordering or its reverse | kernels |
| [`BitonicSort`](@ref AcceleratedKernels.BitonicSort) | no | yes | no | no | all | kernels |
| [`CPUThreads.SampleSort`](@ref AcceleratedKernels.CPUThreads.SampleSort) | yes | yes | yes | yes | all | host |

"Kernels" means GPU backends, and the host backend on KernelAbstractions 0.10, which runs
AcceleratedKernels' kernels on PoCL. See [Algorithms and backends](@ref) for how `Auto` chooses.

```@docs
AcceleratedKernels.MergeSort
AcceleratedKernels.RadixSort
AcceleratedKernels.BitonicSort
AcceleratedKernels.CPUThreads.SampleSort
```

```julia
v = ROCArray(rand(Float32, 1_000_000))
AK.sort!(v; alg=AK.RadixSort())                      # this algorithm, with the device's settings
AK.sort!(v; alg=AK.RadixSort(items_per_thread=4))    # and this setting
AK.sort!(v; alg=AK.Auto(stable=false))               # allow unstable algorithms (Base's QuickSort)
```

Multidimensional arrays are sorted as one flat vector by default; pass `dims` to sort each 1D slice
along that dimension independently, like `Base.sort!(A; dims)`. `sortperm` along `dims` returns
linear indices into the array, so `A[ix]` is sorted along `dims`:
```julia
A = ROCArray(rand(Float32, 1000, 1000))
AK.sort!(A; dims=1)             # each column sorted
ix = AK.sortperm(A; dims=2)     # A[ix] has each row sorted
```

The sorts' scratch memory (the merge and radix sorts' swap buffers, radix sort's histograms) can
be allocated once and reused with a [workspace](workspace.md):
```julia
v = ROCArray(rand(Float32, 100_000))
ws = AK.workspace(AK.sort!, v)
AK.sort!(v; workspace=ws)
```
