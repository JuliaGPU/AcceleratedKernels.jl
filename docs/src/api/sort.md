###  `sort` and friends

Sorting algorithms with similar interface and default settings as the Julia Base ones, on GPUs:
- `sort!` (in-place), `sort` (out-of-place)
- `sortperm!`, `sortperm`
- **Other names**: `sort`, `sort_team`, `sort_team_by_key`, `stable_sort` or variations in Kokkos, RAJA, Thrust that I know of.

Function signatures:
```@docs
AcceleratedKernels.sort!
AcceleratedKernels.sort
AcceleratedKernels.sortperm!
AcceleratedKernels.sortperm
```

Specific implementations that the interfaces above forward to:
- `sample_sort!` - multithreaded CPU sample sort, deferring to Base.sort! on independent slices.
- `merge_sort!` (in-place), `merge_sort` (out-of-place) - sort arbitrary objects with custom comparisons.
- `merge_sort_by_key!`, `merge_sort_by_key` - sort a vector of keys along with a "payload", a vector of corresponding values.
- `merge_sortperm!`, `merge_sortperm`, `merge_sortperm_lowmem!`, `merge_sortperm_lowmem` - compute a sorting index permutation. 

Algorithm choice is available on `sort!` / `sort` / `sortperm!` / `sortperm` with `alg=AK.MergeSort()`,
`alg=AK.MergeSort(lowmem=true)`, `alg=AK.RadixSort()`, or `alg=AK.SampleSort()`, depending on the
backend and operation.

Function signatures:
```@docs
AcceleratedKernels.MergeSort
AcceleratedKernels.RadixSort
AcceleratedKernels.SampleSort
AcceleratedKernels.sample_sort!
AcceleratedKernels.sample_sortperm!
AcceleratedKernels.merge_sort!
AcceleratedKernels.merge_sort
AcceleratedKernels.merge_sort_by_key!
AcceleratedKernels.merge_sort_by_key
AcceleratedKernels.merge_sortperm!
AcceleratedKernels.merge_sortperm
AcceleratedKernels.merge_sortperm_lowmem!
AcceleratedKernels.merge_sortperm_lowmem
```

Example:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32, 100_000))
AK.sort!(v)
```

As GPU memory is more expensive, all functions in AcceleratedKernels.jl expose any temporary arrays they will use (the `temp` argument); you can supply your own buffers to make the algorithms not allocate additional GPU storage, e.g.:
```julia
v = ROCArray(rand(Float32, 100_000))
temp = similar(v)
AK.sort!(v, temp=temp)
```
