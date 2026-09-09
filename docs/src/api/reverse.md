### Reverse

Use `dims` to reverse selected dimensions of an array, or `start` and `stop` to reverse
part of a vector. Either bound may be omitted to use that end of the vector.

```julia
import AcceleratedKernels as AK

v = [1, 2, 3, 4, 5]
AK.reverse!(v; start=2, stop=4)  # v is now [1, 4, 3, 2, 5]
w = AK.reverse(v; start=3)      # w is [1, 4, 5, 2, 3]; v is unchanged
```

```@docs
AcceleratedKernels.reverse!
AcceleratedKernels.reverse
```
