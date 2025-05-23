
import AcceleratedKernels as AK
using Random
Random.seed!(0)

v = rand(1:100, 1_000_000)
AK.sort!(v)
@assert issorted(v)

v = rand(1:100, 1_000_000)
ix = AK.sortperm(v)
@assert issorted(v[ix])


for _ in 1:1000
    num_elems = rand(1:100_000)
    v = array_from_host(rand(Int32, num_elems))
    AK.sample_sort!(v)
    vh = Array(v)
    @assert issorted(vh)
end

