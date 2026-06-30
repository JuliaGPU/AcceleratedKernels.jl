### Binary Search

Find the indices where some elements `x` should be inserted into a sorted sequence `v` to maintain the sorted order. Effectively applying the Julia.Base functions in parallel on a GPU using `foreachindex`.
- `searchsortedfirst!` (in-place), `searchsortedfirst` (allocating): index of first element in `v` >= `x[j]`.
- `searchsortedlast!`, `searchsortedlast`: index of last element in `v` <= `x[j]`.
- **Other names**: `thrust::upper_bound`, `std::lower_bound`.


Example:
```julia
import AcceleratedKernels as AK
using Metal

# Sorted array
v = MtlArray(rand(Float32, 100_000))
AK.sort!(v; alg=AK.MergeSort())

# Elements `x` to place within `v` at indices `ix`
x = MtlArray(rand(Float32, 10_000))
ix = MtlArray{Int}(undef, 10_000)

AK.searchsortedfirst!(ix, v, x)
```


```@docs
AcceleratedKernels.searchsortedfirst!
AcceleratedKernels.searchsortedfirst
AcceleratedKernels.searchsortedlast!
AcceleratedKernels.searchsortedlast
```
