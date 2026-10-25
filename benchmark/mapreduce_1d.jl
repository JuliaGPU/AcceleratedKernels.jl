group = addgroup!(SUITE, "mapreduce_1d")

n = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_1d"] = @benchmarkable @sb(Base.reduce(+, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d"] = @benchmarkable @sb(AK.reduce(+, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))

    T == Float32 || continue

    _group["base_1d_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v, init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
end
