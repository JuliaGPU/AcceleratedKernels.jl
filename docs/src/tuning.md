# Tuning and capabilities (developer notes)

This page is for contributors and for AcceleratedKernels' own package extensions. Nothing here is
public API: the hooks and tuning structs can change in any release, including patch releases.

## Resolution

Every operation resolves its `alg` keyword once, on the host, before touching any data. For
sorting, `resolve_sort` runs three steps:

1. for `Auto`, `select_sort` picks an algorithm from the device's tuning, the backend's
   capabilities and the call's static facts, returning it with unset fields;
2. `fill_defaults` fills the unset fields from the tuning;
3. `check_sort` checks the complete algorithm against the capabilities, the operation and the
   arguments (with `check_fields` for its fields, e.g. a `block_size` that is not a power of
   two), and throws an `ArgumentError` for anything it cannot run.

```@docs
AcceleratedKernels.resolve_sort
```

The other families follow the same steps, with a one-line selection: reductions resolve with
`resolve_reduce`, scans with `resolve_scan`, `findall` with `resolve_findall` and
`any`/`all` with `resolve_predicate`.

```@docs
AcceleratedKernels.resolve_reduce
AcceleratedKernels.resolve_scan
AcceleratedKernels.resolve_findall
AcceleratedKernels.resolve_predicate
```

`Auto` and explicit algorithms share steps 2 and 3, so a tuning cannot make an invalid algorithm
run, and an explicit setting always wins over the tuning.

## Tunings

One plain struct per operation family holds the values that drive selection and fill unset
fields, and one hook returns it for a backend and element type:

```@docs
AcceleratedKernels.SortTuning
AcceleratedKernels.sort_tuning
AcceleratedKernels.ReduceTuning
AcceleratedKernels.reduce_tuning
AcceleratedKernels.ScanTuning
AcceleratedKernels.scan_tuning
AcceleratedKernels.FindallTuning
AcceleratedKernels.findall_tuning
AcceleratedKernels.PredicateTuning
AcceleratedKernels.predicate_tuning
```

AcceleratedKernels defines the hook's generic method; values without a measurement next to them
are AK's earlier defaults. A package extension adds one method for its backend type and may choose
values per device, e.g. from the compute capability of the current CUDA device. Tuning queries run
on the calling task, under its current device, never at load or precompilation time. Record the
measurement behind each value next to it.

The sorting thresholds are crossovers between `MergeSort` and `BitonicSort` or `RadixSort`, timed
on random keys of the element types they apply to. Measure the whole-array thresholds on vectors,
and the slice thresholds along `dims=1` and `dims=2` for several total sizes: the crossovers move
with the number of slices, as few of them leave the device short of work.
`radix_min_slice_len` must hold from `radix_min_len` elements in all, the smallest array along
`dims` it applies to.

## Capabilities

Capabilities are correctness facts about a backend, checked for `Auto` and explicit algorithms
alike; tunings cannot change them.

```@docs
AcceleratedKernels.runs_threads
AcceleratedKernels.runs_kernels
AcceleratedKernels.resolve_backend
```
