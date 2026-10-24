# Reductions with a fixed operator: `sum`, `prod`, `maximum`, `minimum` and `count`.

"""
    sum(src::AbstractArray; kwargs...)

Sum of the elements of an array, with Base's `add_sum`, so that small integers are summed as
`Int`. The keywords are those of [`mapreduce`](@ref). Without `init`, an empty array, or along
`dims` an empty slice, sums to zero of the accumulator type.

```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Int32(1):Int32(100), 100_000))
s = AK.sum(v)                           # an Int
s = AK.sum(v; init=Int32(0))            # an Int as well
s = AK.sum(v; acctype=Int32)            # an Int32, summed as Int32

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
s = AK.sum(m; dims=1)                   # row-wise
```
"""
sum(src::AbstractArray; kwargs...) = _reduce_or_empty(zero, Base.add_sum, src; kwargs...)


"""
    prod(src::AbstractArray; kwargs...)

Product of the elements of an array, with Base's `mul_prod`. The keywords are those of
[`mapreduce`](@ref). Without `init`, an empty array, or along `dims` an empty slice, has the
product one of the accumulator type.

```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32(1):Int32(100), 100_000))
p = AK.prod(v)
p = AK.prod(ROCArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
prod(src::AbstractArray; kwargs...) = _reduce_or_empty(one, Base.mul_prod, src; kwargs...)

# `reduce(op, src; kwargs...)`, except that without `init` an empty reduction gives `empty(A)` of
# the accumulator type `A` (for a whole array, and for every output along `dims`), instead of
# applying it as an `init`, which would change results such as the sign of a sum of `-0.0`s
function _reduce_or_empty(empty, op, src; init=NoInit(), dims=:, acctype=nothing,
                          backend::Union{Nothing, Backend}=nothing, alg::Algorithm=Auto(),
                          workspace=nothing, kwargs...)
    kwargs = (; dims, acctype, backend, alg, kwargs...)
    init isa NoInit || return reduce(op, src; init, workspace, kwargs...)
    A = resolve_acctype(op, Union{}, mapped_eltype(identity, src), acctype)
    if A !== Union{} && isempty(src) &&
       (iswhole(dims) || Base.any(d -> size(src, d) == 0, _reduced_dims(dims, ndims(src))))
        # (the algorithm and the workspace are checked as for any other input)
        p = plan(reduce, op, src; kwargs...)
        scratch_buffers(p, workspace, src)
        iswhole(dims) && return empty(A)
        dims_valid = _reduced_dims(dims, ndims(src))
        dst_sizes = ntuple(d -> d in dims_valid ? 1 : size(src, d), ndims(src))
        return fill!(KernelAbstractions.allocate(p.backend, A, dst_sizes), empty(A))
    end
    return reduce(op, src; workspace, kwargs...)
end


"""
    maximum(src::AbstractArray; kwargs...)

Maximum of the elements of an array; the maximum of an empty array, or along `dims` of an empty
slice, is an error unless `init` is given. The keywords are those of [`mapreduce`](@ref).

```julia
import AcceleratedKernels as AK
using oneAPI

v = oneArray(rand(Int32(1):Int32(100), 100_000))
m = AK.maximum(v)
m = AK.maximum(oneArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
maximum(src::AbstractArray; kwargs...) = reduce(max, src; kwargs...)


"""
    minimum(src::AbstractArray; kwargs...)

Minimum of the elements of an array; the minimum of an empty array, or along `dims` of an empty
slice, is an error unless `init` is given. The keywords are those of [`mapreduce`](@ref).

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Int32(1):Int32(100), 100_000))
m = AK.minimum(v)
m = AK.minimum(CuArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
minimum(src::AbstractArray; kwargs...) = reduce(min, src; kwargs...)


"""
    count([f=identity,] src::AbstractArray; init=0, kwargs...)

Count the elements of `src` for which `f` returns `true`: `f` must return a `Bool`, and the
count is added to `init` (so an empty array counts `init`). The keywords are those of
[`mapreduce`](@ref).

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 100_000))
c = AK.count(x -> x > 0.5, v)
c = AK.count(CuArray(rand(Bool, 10, 100_000)); init=Int32(0), dims=2)
```
"""
count(src::AbstractArray; kwargs...) = count(identity, src; kwargs...)
count(f, src::AbstractArray; init=0, kwargs...) =
    mapreduce(BoolResult(f), Base.add_sum, src; init, kwargs...)

plan(::typeof(sum), src::AbstractArray; kwargs...) = plan(reduce, Base.add_sum, src; kwargs...)
plan(::typeof(prod), src::AbstractArray; kwargs...) = plan(reduce, Base.mul_prod, src; kwargs...)
plan(::typeof(maximum), src::AbstractArray; kwargs...) = plan(reduce, max, src; kwargs...)
plan(::typeof(minimum), src::AbstractArray; kwargs...) = plan(reduce, min, src; kwargs...)
plan(::typeof(count), src::AbstractArray; kwargs...) = plan(count, identity, src; kwargs...)
plan(::typeof(count), f, src::AbstractArray; init=0, kwargs...) =
    plan(mapreduce, BoolResult(f), Base.add_sum, src; init, kwargs...)
