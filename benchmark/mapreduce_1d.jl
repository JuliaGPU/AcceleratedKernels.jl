group = addgroup!(SUITE, "mapreduce_1d")

n = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_1d"] = @benchmarkable @sb(Base.accumulate(+, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))

    T == Float32 || continue

    _group["base_1d_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))

    _group["base_1d_multi"] = @benchmarkable @sb(Base.mapreduce((x, y, z) -> x * (y - z), +, a, b, c, init=$T(0))) setup=(a = ArrayType(rand(rng, $randrange, n)); b = ArrayType(rand(rng, $randrange, n)); c = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d_multi"] = @benchmarkable @sb(AK.mapreduce((x, y, z) -> x * (y - z), +, a, b, c, init=$T(0))) setup=(a = ArrayType(rand(rng, $randrange, n)); b = ArrayType(rand(rng, $randrange, n)); c = ArrayType(rand(rng, $randrange, n)))
end
