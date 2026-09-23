### Algorithms and backends

Every algorithmic operation takes an `alg` keyword. Its default,
[`Auto()`](@ref AcceleratedKernels.Auto), lets AcceleratedKernels choose the algorithm and its
settings; any other value is an algorithm you choose, with the settings you give it.

```@docs
AcceleratedKernels.Algorithm
AcceleratedKernels.Auto
AcceleratedKernels.SortAlgorithm
AcceleratedKernels.CPUThreads
```

The algorithm types of each operation are documented with it, for [sorting](sort.md) and for
[reductions](reduce.md).

#### How `Auto` chooses

`Auto` looks at the backend and its current device, the element type, the layout (a whole array or
slices along `dims`, and their length) and the ordering, never at the array's contents. On the
host backend it chooses the family's `CPUThreads` algorithm, which runs on Julia threads. On a GPU
it follows per-device tuning values: for sorting, `BitonicSort` for short arrays and slices,
`RadixSort` for long whole arrays of the element types it supports, and `MergeSort` otherwise,
subject to `Auto`'s requirements (`stable=true` by default); for reductions, `BlockReduce`.

Algorithms carry their settings as fields, and a field left at `nothing` takes the device's
tuned value:
```julia
AK.sort!(v; alg=AK.RadixSort())                   # the device's block size and items per thread
AK.sort!(v; alg=AK.RadixSort(block_size=512))     # 512 threads, the device's items per thread
```

Explicit algorithms are checked before any data is touched: an algorithm the operation, the
element type, the ordering or the backend cannot run, or an invalid setting, is an
`ArgumentError`. Each new combination of settings compiles new kernels.

#### The `backend` keyword

Operations run on the backend of their arrays: `backend` is derived from every array argument,
the destination first, and all of them must agree. Ranges, `CartesianIndices`, `LinearIndices`
(also through views and reshapes), numbers and other non-array arguments do not count. If no
argument determines the backend (e.g. a loop over a range), the operation runs on the host.

Pass `backend` explicitly for arrays that cannot tell (a range to be processed on a GPU, for
instance), or for memory that several backends can access. AcceleratedKernels does not check that
the backend can reach the arrays. The device and stream are those of the calling task, as set by
the backend package (e.g. `CUDA.device!`); make sure the arrays live on that device.

#### The host backend

Arrays in host memory (`Array` and views of it) are on the host backend. `Auto` processes them on
Julia threads with the `CPUThreads` algorithms. On KernelAbstractions 0.10 the host backend also
runs AcceleratedKernels' GPU kernels (on PoCL), so an explicit kernel algorithm such as
`MergeSort()` works on host arrays there; on KernelAbstractions 0.9 it is an `ArgumentError`.
