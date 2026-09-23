"""
    sum(src::AbstractArray; init=zero(eltype(src)), kwargs...)

Sum of the elements of an array, with optional `init` and `dims`. The keywords are those of
[`reduce`](@ref).

```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Int32(1):Int32(100), 100_000))
s = AK.sum(v)

m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
s = AK.sum(m; dims=1)                   # row-wise
s = AK.sum(m; dims=2, temp=MtlArray(zeros(Int32, 10, 1)))   # into a preallocated result
```
"""
sum(src::AbstractArray; init=zero(eltype(src)), kwargs...) = reduce(+, src; init, kwargs...)


"""
    prod(src::AbstractArray; init=one(eltype(src)), kwargs...)

Product of the elements of an array, with optional `init` and `dims`. The keywords are those of
[`reduce`](@ref).

```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32(1):Int32(100), 100_000))
p = AK.prod(v)
p = AK.prod(ROCArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
prod(src::AbstractArray; init=one(eltype(src)), kwargs...) = reduce(*, src; init, kwargs...)


"""
    maximum(src::AbstractArray; init=typemin(eltype(src)), kwargs...)

Maximum of the elements of an array, with optional `init` and `dims`. The keywords are those of
[`reduce`](@ref).

```julia
import AcceleratedKernels as AK
using oneAPI

v = oneArray(rand(Int32(1):Int32(100), 100_000))
m = AK.maximum(v)
m = AK.maximum(oneArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
maximum(src::AbstractArray; init=typemin(eltype(src)), kwargs...) = reduce(max, src; init, kwargs...)


"""
    minimum(src::AbstractArray; init=typemax(eltype(src)), kwargs...)

Minimum of the elements of an array, with optional `init` and `dims`. The keywords are those of
[`reduce`](@ref).

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Int32(1):Int32(100), 100_000))
m = AK.minimum(v)
m = AK.minimum(CuArray(rand(Int32(1):Int32(100), 10, 100_000)); dims=1)
```
"""
minimum(src::AbstractArray; init=typemax(eltype(src)), kwargs...) = reduce(min, src; init, kwargs...)


"""
    count([f=identity,] src::AbstractArray; init=0, kwargs...)

Count the elements of `src` for which `f` returns `true`, with optional `init` and `dims`; the
result has the type of `init`. The keywords are those of [`mapreduce`](@ref).

```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 100_000))
c = AK.count(x -> x > 0.5, v)
c = AK.count(CuArray(rand(Bool, 10, 100_000)); init=Int32(0), dims=2)
```
"""
function count(src::AbstractArray; init=0, kwargs...)
    mapreduce(x -> x ? one(typeof(init)) : zero(typeof(init)), +, src;
              init, neutral=zero(typeof(init)), kwargs...)
end

function count(f, src::AbstractArray; init=0, kwargs...)
    mapreduce(x -> f(x) ? one(typeof(init)) : zero(typeof(init)), +, src;
              init, neutral=zero(typeof(init)), kwargs...)
end


"""
    cumsum(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=zero(eltype(src)),
        neutral=zero(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # Algorithm choice
        alg::AccumulateAlgorithm=ScanPrefixes(),

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Cumulative sum of elements of an array, with optional `init` and `dims`. Arguments are the same as
for [`accumulate`](@ref).

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
function cumsum(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=zero(eltype(src)),
    neutral=zero(eltype(src)),
    kwargs...
)
    accumulate(
        +, src, backend;
        init, neutral,
        inclusive=true,
        kwargs...
    )
end


"""
    cumprod(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=one(eltype(src)),
        neutral=one(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # Algorithm choice
        alg::AccumulateAlgorithm=ScanPrefixes(),

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Cumulative product of elements of an array, with optional `init` and `dims`. Arguments are the same
as for [`accumulate`](@ref).

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
function cumprod(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=one(eltype(src)),
    neutral=one(eltype(src)),
    kwargs...
)
    accumulate(
        *, src, backend;
        init, neutral,
        inclusive=true,
        kwargs...
    )
end
