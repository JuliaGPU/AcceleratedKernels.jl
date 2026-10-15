group = addgroup!(SUITE, "accumulate_nd")

acc_f(x, y) = sin(x) + cos(y)

GPUArrays.neutral_element(::typeof(acc_f), T) = T(0)

n1 = 3
n2 = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_dims=1"] = @benchmarkable @sb(Base.accumulate(+, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=1"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_dims=2"] = @benchmarkable @sb(Base.accumulate(+, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=2"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    # Many contiguous slices, and many strided ones
    for dims in 1:2
        _group["acck_512x1000_dims=$dims"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0), dims=$dims)) setup=(v = ArrayType(rand(rng, $randrange, 512, 1000)))
    end

    T == Float32 || continue

    _group["base_sincos_dims=1"] = @benchmarkable @sb(Base.accumulate(acc_f, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_sincos_dims=1"] = @benchmarkable @sb(AK.accumulate(acc_f, v, init=$T(0), neutral=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_sincos_dims=2"] = @benchmarkable @sb(Base.accumulate(acc_f, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_sincos_dims=2"] = @benchmarkable @sb(AK.accumulate(acc_f, v, init=$T(0), neutral=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
end
