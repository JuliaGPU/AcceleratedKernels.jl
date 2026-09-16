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

Algorithm choice is available on `sort!` / `sort` / `sortperm!` / `sortperm` with `alg=AK.MergeSort()`,
`alg=AK.MergeSort(lowmem=true)`, `alg=AK.RadixSort()`, `alg=AK.BitonicSort()`, or
`alg=AK.SampleSort()`, depending on the backend and operation.

Function signatures:
```@docs
AcceleratedKernels.MergeSort
AcceleratedKernels.RadixSort
AcceleratedKernels.BitonicSort
AcceleratedKernels.SampleSort
```

Example:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32, 100_000))
AK.sort!(v)
```

Multidimensional arrays are sorted as one flat vector by default; pass `dims` to sort each 1D slice
along that dimension independently, like `Base.sort!(A; dims)`. `sortperm` along `dims` returns
linear indices into the array, so `A[ix]` is sorted along `dims`:
```julia
A = ROCArray(rand(Float32, 1000, 1000))
AK.sort!(A; dims=1)             # each column sorted
ix = AK.sortperm(A; dims=2)     # A[ix] has each row sorted
```

On GPU backends `dims` uses merge sort by default (`RadixSort()` does not support it); on CPU
backends each slice is sorted with `Base.sort!`. `BitonicSort()` is unstable and supports `sort!`
and `sort`, including `dims`, but not `sortperm!` or `sortperm`. It is fastest for small arrays
and short slices, slower than `MergeSort`/`RadixSort` for large whole-array sorts:
```julia
A = ROCArray(rand(Float32, 64, 100_000))
AK.sort!(A; dims=1, alg=AK.BitonicSort())
```

As GPU memory is more expensive, all functions in AcceleratedKernels.jl expose any temporary arrays they will use (the `temp` argument); you can supply your own buffers to make the algorithms not allocate additional GPU storage, e.g.:
```julia
v = ROCArray(rand(Float32, 100_000))
temp = similar(v)
AK.sort!(v, temp=temp)
```
