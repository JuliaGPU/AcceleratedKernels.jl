
import AcceleratedKernels as AK
using BenchmarkTools



v = rand(1_000_000)
init = eltype(v)(0)

r1 = Base.accumulate(+, v; init=init)
r2 = AK.accumulate(+, v; init=init)

@assert r1 == r2


v = rand(1_000_000)
init = eltype(v)(0)

println("1D Benchmark - Base vs. AK")
display(@benchmark Base.accumulate(+, v; init=init))
display(@benchmark AK.accumulate(+, v; init=init))


v = rand(100, 100, 100)
init = eltype(v)(0)

println("3D Benchmark - Base vs. AK")
display(@benchmark Base.accumulate(+, v; init=init, dims=2))
display(@benchmark AK.accumulate(+, v; init=init, dims=2))

