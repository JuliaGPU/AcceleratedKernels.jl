import AcceleratedKernels as AK
using KernelAbstractions

using BenchmarkTools
using Random
Random.seed!(0)


# Choose the Array backend:
#
# using CUDA
# const ArrayType = CuArray
#
# using AMDGPU
# const ArrayType = ROCArray
#
# using oneAPI
# const ArrayType = oneArray
#
# using Metal
# const ArrayType = MtlArray
#
# using OpenCL
# const ArrayType = CLArray
#
const ArrayType = Array


println("Using ArrayType: ", ArrayType)


n1 = 3
n2 = 1_000_000


println("\n===\nBenchmarking accumulate(+, dims=1) on $n1 × $n2 UInt32 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=UInt32(0), dims=1) setup=(v = ArrayType(rand(UInt32(1):UInt32(100), n1, n2))))
display(@benchmark AK.accumulate(+, v, init=UInt32(0), dims=1) setup=(v = ArrayType(rand(UInt32(1):UInt32(100), n1, n2))))

println("\n===\nBenchmarking accumulate(+, dims=2) on $n1 × $n2 UInt32 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=UInt32(0), dims=2) setup=(v = ArrayType(rand(UInt32(1):UInt32(100), n1, n2))))
display(@benchmark AK.accumulate(+, v, init=UInt32(0), dims=2) setup=(v = ArrayType(rand(UInt32(1):UInt32(100), n1, n2))))




println("\n===\nBenchmarking accumulate(+, dims=1) on $n1 × $n2 Int64 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=Int64(0), dims=1) setup=(v = ArrayType(rand(Int64(1):Int64(100), n1, n2))))
display(@benchmark AK.accumulate(+, v, init=Int64(0), dims=1) setup=(v = ArrayType(rand(Int64(1):Int64(100), n1, n2))))

println("\n===\nBenchmarking accumulate(+, dims=2) on $n1 × $n2 Int64 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=Int64(0), dims=2) setup=(v = ArrayType(rand(Int64(1):Int64(100), n1, n2))))
display(@benchmark AK.accumulate(+, v, init=Int64(0), dims=2) setup=(v = ArrayType(rand(Int64(1):Int64(100), n1, n2))))




println("\n===\nBenchmarking accumulate(+, dims=1) on $n1 × $n2 Float32 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=Float32(0), dims=1) setup=(v = ArrayType(rand(Float32, n1, n2))))
display(@benchmark AK.accumulate(+, v, init=Float32(0), dims=1) setup=(v = ArrayType(rand(Float32, n1, n2))))

println("\n===\nBenchmarking accumulate(+, dims=2) on $n1 × $n2 Float32 - Base vs. AK")
display(@benchmark Base.accumulate(+, v, init=Float32(0), dims=2) setup=(v = ArrayType(rand(Float32, n1, n2))))
display(@benchmark AK.accumulate(+, v, init=Float32(0), dims=2) setup=(v = ArrayType(rand(Float32, n1, n2))))




println("\n===\nBenchmarking accumulate((x, y) -> sin(x) + cos(y)), dims=1) on $n1 × $n2 Float32 - Base vs. AK")
display(@benchmark Base.accumulate((x, y) -> sin(x) + cos(y), v, init=Float32(0), dims=1) setup=(v = ArrayType(rand(Float32, n1, n2))))
display(@benchmark AK.accumulate((x, y) -> sin(x) + cos(y), v, init=Float32(0), neutral=Float32(0), dims=1) setup=(v = ArrayType(rand(Float32, n1, n2))))

println("\n===\nBenchmarking accumulate((x, y) -> sin(x) + cos(y)), dims=2) on $n1 × $n2 Float32 - Base vs. AK")
display(@benchmark Base.accumulate((x, y) -> sin(x) + cos(y), v, init=Float32(0), dims=2) setup=(v = ArrayType(rand(Float32, n1, n2))))
display(@benchmark AK.accumulate((x, y) -> sin(x) + cos(y), v, init=Float32(0), neutral=Float32(0), dims=2) setup=(v = ArrayType(rand(Float32, n1, n2))))
