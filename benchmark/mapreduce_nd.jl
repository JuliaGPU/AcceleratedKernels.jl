group = addgroup!(SUITE, "mapreduce_nd")

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    for (suff, (n1, n2)) in (("L", (3, 1_000_000)), ("", (512, 1000)))
        _group["base_dims=1$(suff)"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=1$(suff)"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        _group["base_dims=2$(suff)"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=2$(suff)"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        T == Float32 || continue

        _group["base_dims=1$(suff)_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=1$(suff)_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        _group["base_dims=2$(suff)_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=2$(suff)_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    end
end
