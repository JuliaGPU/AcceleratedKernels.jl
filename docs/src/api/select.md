### `select!` and `select`

The elements at given positions of the sorted array, or of each sorted slice along `dims`, without
sorting it: order statistics such as medians, quantiles or the `k`-th smallest element.
- `select!(dst, src, ranks)` (into `dst`), `select(src, ranks)` (allocating it)

Base's closest function is `partialsort!`, which permutes its input and returns a value or a view;
`select!` leaves its input untouched and writes the selected values, for any number of ranks at
once and along `dims`. C++ calls a single selection `std::nth_element`; CUB has a top-k
(`cub::DeviceTopK`) but no selection by rank.

Function signatures:
```@docs
AcceleratedKernels.select!
AcceleratedKernels.select
```

Example:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 1_000_000))
lo, hi = Array(AK.select(v, [500_000, 500_001]))
median = (lo + hi) / 2

A = CuArray(rand(Float32, 1000, 64))
AK.select(A, [10, 990]; dims=1)              # a 2×64 matrix: the 10th and 990th order statistics of each column
```

[`Auto()`](@ref AcceleratedKernels.Auto), the default `alg`, picks `RadixSelect` on GPUs for the
element types it supports and one or two ranks, and `SortSelect` otherwise and on the host:

| Algorithm | Ranks | Element types, orderings | Scratch | Backends |
|---|---|---|---|---|
| [`RadixSelect`](@ref AcceleratedKernels.RadixSelect) | 1 or 2 | 32/64-bit integers and floats, default ordering or its reverse | none for slices of up to 8192 elements; for longer ones, a buffer of 1/8 of `src` for the remaining candidates, and a few KiB of histograms and extrema per slice | kernels |
| [`SortSelect`](@ref AcceleratedKernels.SortSelect) | any | all | a copy of `src`, and the sort's | all |

`RadixSelect` reads `src` once per 8-bit digit of the keys that the leading digits do not settle,
and slices of up to 8192 elements in a single kernel; for a median of 2^26 `Float32` values it
takes about a tenth of the time of `SortSelect`. For more than two ranks, call `select!` once per
pair, or use `SortSelect` when there are many.

```@docs
AcceleratedKernels.SelectAlgorithm
AcceleratedKernels.RadixSelect
AcceleratedKernels.SortSelect
```
