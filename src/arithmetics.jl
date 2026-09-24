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
function _reduce_or_empty(empty, op, src; init=_NoInit(), dims=:, acctype=nothing,
                          backend::Union{Nothing, Backend}=nothing, alg::Algorithm=Auto(),
                          kwargs...)
    init isa _NoInit || return reduce(op, src; init, dims, acctype, backend, alg, kwargs...)
    A = _acctype(op, Union{}, _mapped_eltype(identity, src), acctype)
    if A !== Union{} && isempty(src)
        # (the algorithm is checked as for any other input)
        b = _resolve_backend(backend, src)
        if _whole(dims)
            _resolve_reduce(alg, b, A, dims)
            return empty(A)
        end
        dims_valid = _reduced_dims(dims, ndims(src))
        if Base.any(d -> size(src, d) == 0, dims_valid)
            _resolve_reduce(alg, b, A, dims_valid)
            dst_sizes = ntuple(d -> d in dims_valid ? 1 : size(src, d), ndims(src))
            return fill!(KernelAbstractions.allocate(b, A, dst_sizes), empty(A))
        end
    end
    return reduce(op, src; dims, acctype, backend, alg, kwargs...)
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
    mapreduce(_BoolValued(f), Base.add_sum, src; init, kwargs...)


"""
    cumsum(src::AbstractArray; kwargs...)

Cumulative sum of elements of an array, as `Base.cumsum` (Base's `add_sum`, so small integers are
summed as `Int`), except that without `dims` a multidimensional array is summed in linear order.
The keywords are those of [`accumulate`](@ref).

# Examples
Simple cumulative sum of elements in a vector:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32(1):Int32(100), 100_000))
s = AK.cumsum(v)
```

Row-wise cumulative sum of a matrix:
```julia
m = ROCArray(rand(Int32(1):Int32(100), 10, 100_000))
s = AK.cumsum(m, dims=1)
```
"""
cumsum(src::AbstractArray; kwargs...) = accumulate(Base.add_sum, src; kwargs...)


"""
    cumprod(src::AbstractArray; kwargs...)

Cumulative product of elements of an array, as `Base.cumprod` (Base's `mul_prod`), except that
without `dims` a multidimensional array is multiplied in linear order. The keywords are those of
[`accumulate`](@ref).

# Examples
Simple cumulative product of elements in a vector:
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneArray(rand(Int32(1):Int32(100), 100_000))
p = AK.cumprod(v)
```

Row-wise cumulative product of a matrix:
```julia
m = oneArray(rand(Int32(1):Int32(100), 10, 100_000))
p = AK.cumprod(m, dims=1)
```
"""
cumprod(src::AbstractArray; kwargs...) = accumulate(Base.mul_prod, src; kwargs...)
