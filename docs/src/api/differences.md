### Differences from Base

AcceleratedKernels' functions with Base's names take the arguments Base's do, but follow
contracts of their own, modelled on GPU libraries such as CUB: `init` is applied once, empty
inputs have no implicit result, and results have one documented type. Base's rules that exist for
Base's own reasons (empty results chosen per operator, `typeof(init)` as the result type along
`dims`, and so on) are left to front-ends: GPUArrays.jl, for example, implements Base's API for
GPU arrays on top of AcceleratedKernels and reproduces Base's results. The differences:

| Call | Base | AcceleratedKernels |
|---|---|---|
| `reduce(op, A)`, `mapreduce(f, op, A)` of an empty `A`, no `init` | `Base.mapreduce_empty`: e.g. `sum(Int[]) == 0`, an error for `maximum` and for most mapped reductions | `ArgumentError` (`sum` and `prod` give zero and one) |
| `mapreduce(f, op, A; dims)` with an empty reduced dimension, no `init` | `Base.reducedim_init`: e.g. `[0 0]` for `x -> x + 1` with `+`, an error for `max` | `ArgumentError` where there are outputs (`sum`, `prod` and `count` give zero or one) |
| a one-element reduction, e.g. `reduce((a, b) -> a + b, [true])` | `true` (`Base.mapreduce_first`) | `1`, the accumulator type |
| the element type of `mapreduce(f, op, A; dims, init)` | `typeof(init)` | the accumulator type: `sum(Int8[1 2]; dims=1, init=Int16(0))` is a `Matrix{Int}` |
| the accumulator type of reductions into an array (`sum!`, reductions along `dims`) | depends on the code path, e.g. on the reduced dimension | one rule, the fold type from `eltype(R)` (see [`mapreducedim!`](@ref AcceleratedKernels.mapreducedim!)) |
| an `init` that is not a neutral element of `op` | outside the contract: `init` must be neutral, and it is unspecified whether it is used for non-empty collections | any value, applied exactly once |
| `reduce(op, A; dims)` for an `op` without `Base.reducedim_init`, such as a closure | `MethodError` | works |
| a non-commutative `op` in a reduction | works (elements keep their order) | unsupported |
