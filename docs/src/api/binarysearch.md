### Binary Search

Find the indices where many elements `x[j]` would be inserted into a sorted sequence `v` to maintain the sorted order: the Julia Base functions applied to each query in parallel. There are no allocating forms, because `Base.searchsortedfirst(v, x)` treats a vector `x` as one value.
- `searchsortedfirst!`: index of the first element of `v` not ordered before `x[j]` (`>= x[j]` by default).
- `searchsortedlast!`: index of the last element of `v` not ordered after `x[j]` (`<= x[j]` by default).
- **Other names**: `thrust::lower_bound`, `thrust::upper_bound`, `std::lower_bound`.


Example:
```julia
import AcceleratedKernels as AK
using Metal

# Sorted array
v = MtlArray(rand(Float32, 100_000))
AK.sort!(v)

# Elements `x` to place within `v` at indices `ix`
x = MtlArray(rand(Float32, 10_000))
ix = MtlArray{Int}(undef, 10_000)

AK.searchsortedfirst!(ix, v, x)
```


```@docs
AcceleratedKernels.searchsortedfirst!
AcceleratedKernels.searchsortedlast!
```
