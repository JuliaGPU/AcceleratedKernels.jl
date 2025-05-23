
import AcceleratedKernels as AK
using BenchmarkTools

using Profile
using PProf

using Random
Random.seed!(0)


# Compile
v = rand(1_000_000)
AK.sort!(v)


# Collect a profile
Profile.clear()
v = rand(1_000_000)
@profile AK.sort!(v)
pprof()


println("Base vs AK sort (Int):")
display(@benchmark Base.sort!(v) setup=(v = rand(1:100, 1_000_000)))
display(@benchmark AK.sort!(v) setup=(v = rand(1:100, 1_000_000)))


println("Base vs AK sort (Float64):")
display(@benchmark Base.sort!(v) setup=(v = rand(Float64, 1_000_000)))
display(@benchmark AK.sort!(v) setup=(v = rand(Float64, 1_000_000)))
