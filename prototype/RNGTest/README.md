# AK + RNGTest SmallCrush Prototype

This folder provides a chunked random stream generator based on `AcceleratedKernels.jl` that can be fed into `RNGTest.jl`.

The stream is deterministic and effectively unbounded:
- each refill generates `chunk` random `UInt64` values with `AK.rand!`
- each refill advances one persistent `CounterRNG` stream offset
- this is a practical chunked stream for RNGTest callback mode

`RNGTest.jl` (in this local checkout) expects a callback returning `Float64` in `[0,1]`, so `UInt64` words are mapped to `Float64` via top-53-bit scaling.

Current status in this harness: `SplitMix64`, `Philox`, and `Threefry` all pass BigCrush using `run_bigcrush.jl`.

## Run SmallCrush

From this directory:

```powershell
julia --project=. run_smallcrush.jl
```

## Run BigCrush

```powershell
julia --project=. run_bigcrush.jl
```

Notes:
- Configure `ALG`, `SEED`, and `CHUNK` at the top of
  `run_smallcrush.jl` / `run_bigcrush.jl`.
- The stream refills directly into host scratch using `AK.rand!` on CPU.
- `chunk` controls refill amortization and memory usage.
- `chunk=100000000` means ~800 MB host scratch (`UInt64`).
