group = addgroup!(SUITE, "accumulate_nd")

acc_f(x, y) = sin(x) + cos(y)

GPUArrays.neutral_element(::typeof(acc_f), T) = T(0)

n1 = 3
n2 = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    for (suff, (n1, n2)) in (("L", (3, 1_000_000)), ("", (512, 1000)))
        _group["base_dims=1$suff"] = @benchmarkable @sb(Base.accumulate(+, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=1$suff"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        _group["base_dims=2$suff"] = @benchmarkable @sb(Base.accumulate(+, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_dims=2$suff"] = @benchmarkable @sb(AK.accumulate(+, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        T == Float32 || continue

        _group["base_sincos_dims=1$suff"] = @benchmarkable @sb(Base.accumulate(acc_f, v, init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_sincos_dims=1$suff"] = @benchmarkable @sb(AK.accumulate(acc_f, v, init=$T(0), neutral=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

        _group["base_sincos_dims=2$suff"] = @benchmarkable @sb(Base.accumulate(acc_f, v, init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
        _group["acck_sincos_dims=2$suff"] = @benchmarkable @sb(AK.accumulate(acc_f, v, init=$T(0), neutral=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    end
end
