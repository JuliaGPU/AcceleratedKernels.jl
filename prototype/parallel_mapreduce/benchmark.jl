
import AcceleratedKernels as AK
using BenchmarkTools


v = rand(1_000_000)
f(x) = x^2
op(x, y) = x + y
init = eltype(v)(0)

println("1D Benchmark - Base vs. AK")
display(@benchmark Base.mapreduce(f, op, v; init=init))
display(@benchmark AK.mapreduce(f, op, v; init=init, neutral=init))


v = rand(100, 100, 100)
f(x) = x^2
op(x, y) = x + y
init = eltype(v)(0)

println("3D Benchmark - Base vs. AK")
display(@benchmark Base.mapreduce(f, op, v; init=init, dims=2))
display(@benchmark AK.mapreduce(f, op, v; init=init, neutral=init, dims=2))

