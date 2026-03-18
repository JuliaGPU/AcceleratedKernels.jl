### Random Number Generation

Counter-based random generation for CPU and GPU backends with deterministic behavior for fixed
`seed`, algorithm, array shape, and eltype.

Use an explicit `CounterRNG(seed; alg=...)` when reproducibility matters. For convenience,
`AK.rand!(x)` creates a fresh `CounterRNG()` on each call using one auto-seeded
`Random.rand(Random.default_rng(), UInt64)` draw, so repeated calls produce different outputs unless Random.seed!() is used.

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
- Published/reference versions of `SplitMix64`, `Philox`, and `Threefry` are reported to pass
  TestU01 BigCrush.
- These results refer to specific constructions and test setups; wrapper choices (such as
  seed/key mapping conventions) can change bitwise output streams.
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

`Philox` is the default algorithm for `CounterRNG()`, as it is very thorough and very fast; it has been measured on par with `CUDA.rand!` and `SplitMix64` at ~390 GB/s on an Nvidia GeForce RTX
5060 (advertised 448 GB/s), i.e. effectively memory-bound throughput.

Examples:
```julia
import AcceleratedKernels as AK
using oneAPI

# Reproducible
rng = AK.CounterRNG(0x12345678; alg=AK.Philox())
x = oneArray{Float32}(undef, 1024)
AK.rand!(rng, x)

# Convenience (fresh auto-seeded RNG on each call)
y = oneArray{Float32}(undef, 1024)
AK.rand!(y)
```

```@docs
AcceleratedKernels.CounterRNG
AcceleratedKernels.rand!
```
