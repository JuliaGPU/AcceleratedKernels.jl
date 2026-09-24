# Tuning and capabilities (developer notes)

This page is for contributors and for AcceleratedKernels' own package extensions. Nothing here is
public API: the hooks and tuning structs can change in any release, including patch releases.

## Resolution

Every operation resolves its `alg` keyword once, on the host, before touching any data. For
sorting, `_resolve_sort` runs four steps:

1. `_checkdomain` checks the fields the caller set (e.g. a `block_size` that is not a power of
   two), before any arithmetic uses them;
2. for `Auto`, `_select_sort` picks an algorithm from the device's tuning, the backend's
   capabilities and the call's static facts, returning it with unset fields;
3. `_fill` fills the unset fields from the tuning;
4. `_check` checks the complete algorithm against the capabilities, the operation and the
   arguments, and throws an `ArgumentError` for anything it cannot run.

```@docs
AcceleratedKernels._resolve_sort
```

The other families follow the same steps, with a one-line selection: reductions resolve with
`_resolve_reduce`, scans with `_resolve_scan`, `findall` with `_resolve_findall` and
`any`/`all` with `_resolve_predicate`.

```@docs
AcceleratedKernels._resolve_reduce
AcceleratedKernels._resolve_scan
AcceleratedKernels._resolve_findall
AcceleratedKernels._resolve_predicate
```

`Auto` and explicit algorithms share steps 3 and 4, so a tuning cannot make an invalid algorithm
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

AcceleratedKernels defines the hook's generic method, whose values reproduce the library's
historical defaults. A package extension adds one method for its backend type and may choose
values per device, e.g. from the compute capability of the current CUDA device. Tuning queries run
on the calling task, under its current device, never at load or precompilation time. Record the
measurement behind each value (see `benchmark/tune_sort.jl`) next to it.

## Capabilities

Capabilities are correctness facts about a backend, checked for `Auto` and explicit algorithms
alike; tunings cannot change them.

```@docs
AcceleratedKernels._runs_threads
AcceleratedKernels._runs_kernels
AcceleratedKernels._supports_lookback
AcceleratedKernels._supports_concurrent_write
AcceleratedKernels._resolve_backend
```
