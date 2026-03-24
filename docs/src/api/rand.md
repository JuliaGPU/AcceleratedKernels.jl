### Random Number Generation

Counter-based random generation for CPU and GPU backends with deterministic stream behavior for
fixed `seed`, algorithm, and call sequence.

`CounterRNG` carries an internal `offset` (starting at `0`) that advances by `length(v)` on each
`AK.rand!(rng, v)` call. This means chunked fills are stream-consistent:
- filling `100` then `100` elements yields the same `200` values as one `200`-element fill.
- calls that share the same `CounterRNG` instance concurrently are not thread-safe.
- call `AK.reset!(rng)` to rewind a mutable offset-bearing RNG back to offset `0x0`.

`AK.rand!` also accepts custom `CounterRNG` implementations:
- if they have a mutable `offset` field, streaming advancement is applied
- if they have no `offset` field, each call behaves statelessly from counter `0`
- if they have an immutable `offset` field, that offset is used as a fixed start and is not advanced

Use an explicit `CounterRNG` when reproducibility is required. For
convenience,
`AK.rand!(v)` creates a fresh `CounterRNG()` on each call using one auto-seeded
`Base.rand(UInt64)` draw, so repeated calls produce different outputs unless Random.seed!() is used.

`AK.reset!(rng)` rewinds offset to `0x0` for mutable RNGs that have an `offset` field.

Custom RNGs:
- Define an algorithm type `MyAlg <: AK.CounterRNGAlgorithm`.
- Define a `CounterRNG` with fields `seed` and `alg`.
- Add a mutable `offset::UInt64` field if you want stream advancement across calls; omit it for stateless calls from counter `0`.
- Implement typed `rand_uint` methods:
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt32})::UInt32`
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt64})::UInt64`

Both widths should be implemented so `AK.rand!` supports all integer/float output types without falling back or error.

Supported element types:
- `UInt8`, `UInt16`, `UInt32`, `UInt64`
- `Int8`, `Int16`, `Int32`, `Int64`
- `Float16`, `Float32`, `Float64`
- `Bool`

The core of the random number generation produces either a `UInt32` or `UInt64` depending on the width of the requested element type.
That `UInt` is then either:
- Unsigned integers: returned as-is or truncated if necessary.
- Signed integers: reinterpreted as a signed integer bit pattern and truncated if necessary.
- Floats: mantissa construction into a uniform grid in `[0, 1)` ([read more](https://lomont.org/posts/2017/unit-random/)).
- Bool: `true` if the `UInt` draw is odd (`isodd(u)`), otherwise `false`.

Algorithms currently available:
- `SplitMix64` ([read more](https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64))
- `Philox` ([read more](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf))
- `Threefry` ([read more](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf))

Statistical-testing note:
- In this repository, `SplitMix64`, `Philox`, and `Threefry` have passed TestU01 BigCrush
- These generators are not intended to be cryptographically secure.

Philox keying note:
- AK uses `Philox2x32` internally (one 32-bit Philox key word).
- Users can pass any non-negative `Integer` seed; AK normalises to `UInt64` then derives the
  32-bit Philox key via a SplitMix-based mapping.
- This is a deliberate wrapper choice for ease of use (simple `seed` API with deterministic
  streams), not a change to the Philox round function itself.
- Therefore, AK Philox streams are deterministic and high-quality, but not guaranteed to be
  bit-for-bit identical to a raw Random123 Philox stream unless the same seed-to-key mapping and
  counter convention are used.

`Philox` is the default algorithm for `CounterRNG()` because it is thorough and very fast; it has been measured on par with `CUDA.rand!` and `SplitMix64` at ~390 GB/s on an Nvidia GeForce RTX
5060 (advertised 448 GB/s), i.e. effectively memory-bound throughput.

Examples:
```julia
import AcceleratedKernels as AK
using oneAPI

# Reproducible
rng = AK.CounterRNG(0x12345678; alg=AK.Philox())
v = oneArray{Float32}(undef, 1024)
AK.rand!(rng, v)

# Stream-consistent chunking
v1 = oneArray{Float32}(undef, 100)
v2 = oneArray{Float32}(undef, 100)
AK.rand!(rng, v1)
AK.rand!(rng, v2)

# Convenience (fresh auto-seeded RNG on each call)
y = oneArray{Float32}(undef, 1024)
AK.rand!(y)
```

```@docs
AcceleratedKernels.CounterRNG
AcceleratedKernels.CounterRNGAlgorithm
AcceleratedKernels.reset!
AcceleratedKernels.rand!
```
