group = addgroup!(SUITE, "statistics")

n1d = 1_000_000
n3d = 100
for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "mean $T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["mean1d_statistics"] = @benchmarkable @sb(Statistics.mean(v)) setup=(v = ArrayType(rand(rng, $randrange, n1d)))
    _group["mean1d_ak"] = @benchmarkable @sb(AK.mean(v)) setup=(v = ArrayType(rand(rng, $randrange, n1d)))

    
    _group["meannd_statistics"] = @benchmarkable @sb(Statistics.mean(v,dims=3)) setup=(v = ArrayType(rand(rng, $randrange, n3d,n3d,n3d)))
    _group["meannd_ak"] = @benchmarkable @sb(AK.mean(v,dims=3)) setup=(v = ArrayType(rand(rng, $randrange, n3d,n3d,n3d)))

    local _group = addgroup!(group, "var $T")
    _group["var1d_statistics"] = @benchmarkable @sb(Statistics.var(v)) setup=(v = ArrayType(rand(rng, $randrange, n1d)))
    _group["var1d_ak"] = @benchmarkable @sb(AK.var(v)) setup=(v = ArrayType(rand(rng, $randrange, n1d)))

    
    _group["varnd_statistics"] = @benchmarkable @sb(Statistics.var(v,dims=3)) setup=(v = ArrayType(rand(rng, $randrange, n3d,n3d,n3d)))
    _group["varnd_ak"] = @benchmarkable @sb(AK.var(v,dims=3)) setup=(v = ArrayType(rand(rng, $randrange, n3d,n3d,n3d)))

end
