### Scratch Memory

Many operations need scratch memory: merge and radix sorts swap between two buffers, reductions
keep partial results per block, scans keep the totals of their tiles, and `findall` counts the
selected elements of each block. By default each call allocates what it needs. To allocate it once
and reuse it, make a workspace for the call and pass it through the operation's `workspace`
keyword:

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 1_000_000))
ws = AK.workspace(AK.sort!, v)          # the same arguments as the call
for _ in 1:10
    rand!(v)
    AK.sort!(v; workspace=ws)           # allocates no scratch memory
end

AK.workspace_size(AK.sum, v)            # what the call needs, without allocating it
```

A workspace belongs to one kind of call: the operation resolves its algorithm and computes its
buffers as it would without one, and a workspace made for another backend or device, another
algorithm (`Auto()` may choose differently for another length) or other buffer sizes is an
`ArgumentError`. It holds no state between calls, so calls may reuse it in turn, but not at the
same time on several tasks or streams.

```@docs
AcceleratedKernels.workspace
AcceleratedKernels.workspace_size
AcceleratedKernels.Workspace
```
