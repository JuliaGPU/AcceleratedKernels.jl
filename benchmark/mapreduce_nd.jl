group = addgroup!(SUITE, "mapreduce_nd")

n1 = 3
n2 = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_dims=1"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=1"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_dims=2"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=2"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    T == Float32 || continue

    _group["base_dims=1_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=1_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_dims=2_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=2_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
end

# A fused two-input source, which has no dense buffer: into one output, and into few outputs of
# many elements each
let _group = addgroup!(group, "fused"), n = 128
    for dims in ((1, 2, 3), (1, 2))
        _group["base_dims=$dims"] = @benchmarkable @sb(Base.mapreduce(*, +, v, w; dims=$dims)) setup=(v = ArrayType(rand(rng, Float32, $n, $n, $n)); w = ArrayType(rand(rng, Float32, $n, $n, $n)))
        _group["acck_dims=$dims"] = @benchmarkable @sb(AK.mapreduce(*, +, v, w; dims=$dims)) setup=(v = ArrayType(rand(rng, Float32, $n, $n, $n)); w = ArrayType(rand(rng, Float32, $n, $n, $n)))
    end
end
