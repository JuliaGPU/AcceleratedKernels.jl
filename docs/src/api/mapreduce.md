### MapReduce

Equivalent to `reduce(op, map(f, iterable))`, without saving the intermediate mapped collection; can be used to e.g. split documents into words (map) and count the frequency thereof (reduce).
- **Other names**: `transform_reduce`, some `fold` implementations include the mapping function too.

---

```@docs
AcceleratedKernels.mapreduce
```

To reduce into an existing array, for example to accumulate into it across calls or to avoid an
allocation, use `mapreducedim!`. Its docstring states the contract every reduction follows:
the operator algebra, neutral elements, the accumulator type, how `init` and the destination's
values combine with the result, and empty slices.

```@docs
AcceleratedKernels.mapreducedim!
```
