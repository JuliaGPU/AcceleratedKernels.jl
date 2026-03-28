### Random Number Generation

Counter-based random generation for CPU and GPU backends with deterministic stream behaviour for a
fixed `seed`, algorithm, and call sequence.

Both in-place and allocation forms are supported:
- Uniform: `AK.rand!`, `AK.rand`
- Standard normal: `AK.randn!`, `AK.randn`

`CounterRNG` stores:
- `seed::UInt64`
- algorithm `alg`
- stream `offset::UInt64`

The offset starts at `0` and advances by the number of generated values after each call. For
`AK.rand!(rng, v)` and `AK.randn!(rng, v)`, element `v[i]` is generated from logical counter
`rng.offset + UInt64(i - 1)` in linear indexing order.

This gives stream-consistent chunking:
- filling `100` then `100` elements yields the same `200` values as one `200`-element fill.
- `AK.reset!(rng)` rewinds `rng.offset` to `0x0`.

Calls that share the same `CounterRNG` instance concurrently are not thread-safe and may race on
`offset`.

`AK.rand!` and `AK.randn!` accept `rng::AK.CounterRNG`. Passing other RNG container types is not
supported and will throw a `MethodError`.

#### Auto-seeded convenience behaviour

Use an explicit `CounterRNG` when reproducibility is required.

For convenience, calls without an explicit `rng` construct a fresh `CounterRNG()` on each call,
using one auto-seeded `Base.rand(UInt64)` draw. Therefore repeated bare calls intentionally produce
different outputs unless `Random.seed!()` is used first.

Examples:
- `AK.rand!(v)`
- `AK.randn!(v)`
- `AK.rand(backend, args...)`
- `AK.randn(backend, args...)`

These do **not** continue a shared stream across calls unless you pass the same explicit
`CounterRNG`.

#### Allocation forms

Canonical forms:
- `AK.rand(rng, backend, T, dims...)`
- `AK.randn(rng, backend, T, dims...)`

Shared defaults:
- omit `rng` -> fresh `CounterRNG()`
- omit `backend` -> CPU backend
- omit `T` -> `Float64` on CPU backend, `Float32` otherwise

Common shorthands include:
- `AK.rand(dims...)`
- `AK.rand(T, dims...)`
- `AK.rand(backend, dims...)`
- and the corresponding `AK.randn(...)` variants

For explicit `rng`, both `AK.rand` and `AK.randn` advance `rng.offset` by the number of generated
elements, i.e. `prod(dims)`.

#### Supported element types

`AK.rand!` / `AK.rand` support:
- `UInt8`, `UInt16`, `UInt32`, `UInt64`
- `Int8`, `Int16`, `Int32`, `Int64`
- `Float16`, `Float32`, `Float64`
- `Bool`

`AK.randn!` / `AK.randn` currently support:
- `Float16`, `Float32`, `Float64`

#### Value generation semantics

The core generator produces either a `UInt32` or `UInt64`, depending on the requested output type.
That raw unsigned value is then mapped as follows:
- Unsigned integers: returned directly, or truncated if narrower
- Signed integers: the corresponding unsigned bit pattern reinterpreted as signed, then truncated if narrower
- Floats: mantissa construction onto a uniform grid in `[0, 1)` ([read more](https://lomont.org/posts/2017/unit-random/))
- Bool: `true` if the raw `UInt` draw is odd (`isodd(u)`), otherwise `false`

`AK.randn!` uses Box-Muller with midpoint-mapped open-interval uniforms in `(0, 1)`.

#### Algorithms currently available

- `SplitMix64` ([read more](https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64))
- `Philox` ([read more](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf))
- `Threefry` ([read more](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf))

`Philox` is the default algorithm for `CounterRNG()`.

#### Statistical testing and security

- In this repository, `SplitMix64`, `Philox`, and `Threefry` have passed TestU01 BigCrush
- These generators are not intended to be cryptographically secure

#### Philox keying note

AK uses `Philox2x32` internally, which has a single 32-bit Philox key word.

Users may pass any non-negative `Integer` seed with `seed <= typemax(UInt64)`; AK converts it to
`UInt64` and derives the 32-bit Philox key using SplitMix. This wrapper choice is deliberate for
ease of use and deterministic streams, not a change to the Philox round function itself.

Therefore, AK Philox streams are deterministic and high-quality, but are not guaranteed to be
bit-for-bit identical to a raw Random123 Philox stream unless the same seed-to-key mapping and
counter convention are used.

#### Custom algorithms

To define a custom counter RNG:
- define an algorithm type `MyAlg <: AK.CounterRNGAlgorithm`
- implement:
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt32})::UInt32`
  - `AK.rand_uint(seed::UInt64, alg::MyAlg, counter::UInt64, ::Type{UInt64})::UInt64`

Then use it via:
- `AK.CounterRNG(seed; alg=MyAlg(), offset=...)`

Both widths should be implemented so `AK.rand!` supports all integer and floating-point output
types without fallback or error.

#### Performance note

`Philox` is the default because it is high-quality and very fast. `AK.rand!` has been measured at
roughly memory-bound throughput (~390 GB/s) on an Nvidia GeForce RTX 5060, including slightly better
performance than CURAND for large `CuArray{Float32}` fills and substantially faster `CuArray{Int32}`
filling than native `CUDA.rand!` in the benchmarks used for this repository.

Examples:
```julia
import AcceleratedKernels as AK
using oneAPI
using AMDGPU

# Reproducible
rng = AK.CounterRNG(0x12345678; alg=AK.Philox())
v = oneArray{Float32}(undef, 1024)
AK.rand!(rng, v)

# Stream-consistent chunking
rng = AK.CounterRNG(0x12345678; alg=AK.Philox())
v1 = oneArray{Float32}(undef, 100)
v2 = oneArray{Float32}(undef, 100)
AK.rand!(rng, v1)
AK.rand!(rng, v2)

# Convenience: fresh auto-seeded RNG on each call
y = oneArray{Float32}(undef, 1024)
AK.rand!(y)

# Allocation form
y_cpu_auto = AK.rand(1024)                            # CPU, Vector{Float64}
y_one = AK.rand(oneAPIBackend(), Float32, 1024)       # fresh RNG, allocate + fill oneArray
y_cpu_typed = AK.rand(rng, Float16, 1024)             # CPU backend, explicit type, explicit RNG

# Standard normal filling
z = ROCArray{Float32}(undef, 1024)
AK.randn!(rng, z)

# Standard normal allocation form
z_cpu_auto = AK.randn(1024)                           # CPU, Vector{Float64}
z_roc = AK.randn(ROCBackend(), 1024)                  # fresh RNG, allocate + fill ROCArray{Float32}
z_cpu_typed = AK.randn(rng, Float16, 1024)            # CPU backend, explicit type, explicit RNG
```

```@docs
AcceleratedKernels.CounterRNG
AcceleratedKernels.CounterRNGAlgorithm
AcceleratedKernels.rand_uint
AcceleratedKernels.reset!
AcceleratedKernels.rand!
AcceleratedKernels.rand
AcceleratedKernels.randn!
AcceleratedKernels.randn
```