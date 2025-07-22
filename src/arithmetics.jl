"""
    sum(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=zero(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Sum of elements of an array, with optional `init` and `dims`. Arguments are the same as for
[`reduce`](@ref).

# Examples
Simple sum of elements in a vector:
```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Int32(1):Int32(100), 100_000))
s = AK.sum(v)
```

Row-wise sum of a matrix:
```julia
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
s = AK.sum(m, dims=1)
```

If you know the shape of the resulting array (in case of a axis-wise sum, i.e. `dims` is not
`nothing`), you can provide the `temp` argument to save results into and avoid allocations:
```julia
m = MtlArray(rand(Int32(1):Int32(100), 10, 100_000))
temp = MtlArray(zeros(Int32, 10))
s = AK.sum(m, dims=2, temp=temp)
```
"""
function sum(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=zero(eltype(src)),
    kwargs...
)
    reduce(
        +, src, backend;
        init,
        kwargs...
    )
end


"""
    prod(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=one(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Product of elements of an array, with optional `init` and `dims`. Arguments are the same as for
[`reduce`](@ref).

# Examples
Simple product of elements in a vector:
```julia
import AcceleratedKernels as AK
using AMDGPU

v = ROCArray(rand(Int32(1):Int32(100), 100_000))
p = AK.prod(v)
```

Row-wise product of a matrix:
```julia
m = ROCArray(rand(Int32(1):Int32(100), 10, 100_000))
p = AK.prod(m, dims=1)
```

If you know the shape of the resulting array (in case of a axis-wise product, i.e. `dims` is not
`nothing`), you can provide the `temp` argument to save results into and avoid allocations:
```julia
m = ROCArray(rand(Int32(1):Int32(100), 10, 100_000))
temp = ROCArray(ones(Int32, 10))
p = AK.prod(m, dims=2, temp=temp)
```
"""
function prod(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=one(eltype(src)),
    kwargs...
)
    reduce(
        *, src, backend;
        init,
        kwargs...
    )
end


"""
    maximum(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=typemin(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Maximum of elements of an array, with optional `init` and `dims`. Arguments are the same as for
[`reduce`](@ref).

# Examples
Simple maximum of elements in a vector:
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneArray(rand(Int32(1):Int32(100), 100_000))
m = AK.maximum(v)
```

Row-wise maximum of a matrix:
```julia
m = oneArray(rand(Int32(1):Int32(100), 10, 100_000))
m = AK.maximum(m, dims=1)
```

If you know the shape of the resulting array (in case of a axis-wise maximum, i.e. `dims` is not
`nothing`), you can provide the `temp` argument to save results into and avoid allocations:
```julia
m = oneArray(rand(Int32(1):Int32(100), 10, 100_000))
temp = oneArray(zeros(Int32, 10))
m = AK.maximum(m, dims=2, temp=temp)
```
"""
function maximum(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=typemin(eltype(src)),
    kwargs...
)
    reduce(
        max, src, backend;
        init,
        kwargs...
    )
end


"""
    minimum(
        src::AbstractArray, backend::Backend=get_backend(src);
        init=typemax(eltype(src)),
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Minimum of elements of an array, with optional `init` and `dims`. Arguments are the same as for
[`reduce`](@ref).

# Examples
Simple minimum of elements in a vector:
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Int32(1):Int32(100), 100_000))
m = AK.minimum(v)
```

Row-wise minimum of a matrix:
```julia
m = CuArray(rand(Int32(1):Int32(100), 10, 100_000))
m = AK.minimum(m, dims=1)
```

If you know the shape of the resulting array (in case of a axis-wise minimum, i.e. `dims` is not
`nothing`), you can provide the `temp` argument to save results into and avoid allocations:
```julia
m = CuArray(rand(Int32(1):Int32(100), 10, 100_000))
temp = CuArray(ones(Int32, 10))
m = AK.minimum(m, dims=2, temp=temp)
```
"""
function minimum(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=typemax(eltype(src)),
    kwargs...
)
    reduce(
        min, src, backend;
        init,
        kwargs...
    )
end


"""
    count(
        [f=identity], src::AbstractArray, backend::Backend=get_backend(src);
        init=0,
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Count the number of elements in `src` for which the function `f` returns `true`. If `f` is omitted,
count the number of `true` elements in `src`. Arguments are the same as for [`mapreduce`](@ref).

# Examples
Simple count of `true` elements in a vector:
```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Bool, 100_000))
c = AK.count(v)
```

Count of elements greater than 50 in a vector:
```julia
v = MtlArray(rand(Int32(1):Int32(100), 100_000))
c = AK.count(x -> x > 50, v)
```

Row-wise count of `true` elements in a matrix:
```julia
m = MtlArray(rand(Bool, 10, 100_000))
c = AK.count(m, dims=1)
```

If you know the shape of the resulting array (in case of a axis-wise count, i.e. `dims` is not
`nothing`), you can provide the `temp` argument to save results into and avoid allocations:
```julia
m = MtlArray(rand(Bool, 10, 100_000))
temp = MtlArray(zeros(Int32, 10))
c = AK.count(m, dims=2, temp=temp)
```
"""
function count(
    src::AbstractArray, backend::Backend=get_backend(src);
    init=0,
    kwargs...
)
    mapreduce(
        x -> x ? one(typeof(init)) : zero(typeof(init)), +, src, backend;
        init,
        neutral=zero(typeof(init)),
        kwargs...
    )
end


function count(
    f, src::AbstractArray, backend::Backend=get_backend(src);
    init=0,
    kwargs...
)
    mapreduce(
        x -> f(x) ? one(typeof(init)) : zero(typeof(init)), +, src, backend;
        init,
        neutral=zero(typeof(init)),
        kwargs...
    )
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
