### Random Number Generation

Counter-based random generation for CPU and GPU backends with deterministic stream behavior for
fixed `seed`, algorithm, and call sequence.

Both in-place and allocation forms are supported:
- Uniform: `AK.rand!`, `AK.rand`
- Standard normal: `AK.randn!`, `AK.randn`

`CounterRNG` carries an internal `offset` (starting at `0`) that advances by `length(v)` on each
`AK.rand!(rng, v)` call. This means chunked fills are stream-consistent:
- filling `100` then `100` elements yields the same `200` values as one `200`-element fill.
- calls that share the same `CounterRNG` instance concurrently are not thread-safe.
- call `AK.reset!(rng)` to rewind a `CounterRNG` offset back to `0x0`.

`AK.rand!(rng, v)` accepts `rng::AK.CounterRNG`.
Passing other RNG container types is not supported and will throw a `MethodError`.

Use an explicit `CounterRNG` when reproducibility is required. For
convenience,
`AK.rand!(v)` creates a fresh `CounterRNG()` on each call using one auto-seeded
`Base.rand(UInt64)` draw, so repeated calls produce different outputs unless Random.seed!() is used.
Likewise, `AK.rand(backend, args...)` creates a fresh auto-seeded `CounterRNG()` on each call.

`AK.reset!(rng::AK.CounterRNG)` rewinds `rng.offset` to `0x0`.

Allocation convenience:
- Canonical forms are `AK.rand(rng, backend, T, dims...)` and `AK.randn(rng, backend, T, dims...)`.
- Defaults are shared: omit `rng` -> fresh `CounterRNG()`; omit `backend` -> CPU backend; omit `T` -> `Float64` on CPU backend and `Float32` otherwise.
- Common shorthands include `AK.rand(dims...)`, `AK.rand(T, dims...)`, `AK.rand(backend, dims...)`, and the corresponding `AK.randn(...)` variants.
- For explicit `rng`, both `AK.rand` and `AK.randn` advance `rng.offset` by `prod(dims)`.

Custom algorithms:
- Define an algorithm type `MyAlg <: AK.CounterRNGAlgorithm`.
- Implement typed `rand_uint` methods:
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt32})::UInt32`
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt64})::UInt64`
- Use your algorithm via `AK.CounterRNG(seed; alg=MyAlg(), offset=...)`.

Both widths should be implemented so `AK.rand!` supports all integer/float output types without falling back or error.

Supported element types:
- `UInt8`, `UInt16`, `UInt32`, `UInt64`
- `Int8`, `Int16`, `Int32`, `Int64`
- `Float16`, `Float32`, `Float64`
- `Bool`

`AK.randn!` fills arrays with standard normal samples and currently supports:
- `Float16`, `Float32`, `Float64`

`AK.randn!` uses Box-Muller with open-interval uniforms in `(0, 1)` from a branch-free midpoint mapping.

`AK.randn!(v)` and `AK.randn(backend, args...)` create a fresh auto-seeded `CounterRNG()` on each
call, so repeated calls produce different outputs unless `Random.seed!()` is used.

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
using ROCArray

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

# Allocation form
y_cpu_auto = AK.rand(1024)                               # defaults to CPU, Vector{Float64}
y_oneArray = AK.rand(oneAPIBackend(), Float32, 1024)     # fresh RNG, allocate and fill oneArray
y_cpu_typed = AK.rand(rng, Float16, 1024)                # CPU backend, explicit type, explicit RNG

# Standard normal filling
z = ROCArray{Float32}(undef, 1024)
AK.randn!(rng, z)

# Standard normal allocation form
z_cpu_auto = AK.randn(1024)                              # defaults to CPU, Vector{Float64}
z_ROCArray = AK.randn(oneAPIBackend(), 1024)             # allocate and fill ROCArray{Float32}
z_cpu_typed = AK.randn(rng, Float16, 1024)               # CPU backend, explicit type, explicit RNG
```

```@docs
AcceleratedKernels.CounterRNG
AcceleratedKernels.CounterRNGAlgorithm
AcceleratedKernels.reset!
AcceleratedKernels.rand!
AcceleratedKernels.rand
AcceleratedKernels.randn!
AcceleratedKernels.randn
```
