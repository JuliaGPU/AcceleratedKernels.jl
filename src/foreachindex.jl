@kernel inbounds=true cpu=false unsafe_indices=true function _forindices_global!(f, indices)

    # Calculate global index
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N

    if i <= length(indices)
        f(indices[i])
    end
end


function _forindices_gpu(
    f,
    indices,
    backend::Backend;

    block_size::Int=256,
)
    # GPU implementation
    @argcheck block_size > 0
    blocks = (length(indices) + block_size - 1) ÷ block_size
    _forindices_global!(backend, block_size)(f, indices, ndrange=(block_size * blocks,))
    nothing
end


function _forindices_threads(f, indices; max_tasks, min_elems)
    task_partition(length(indices), max_tasks, min_elems) do irange
        # Task partition returns static ranges indexed from 1:length(indices); use those to index
        # into indices, which supports arbitrary indices (and gets compiled away when using 1-based
        # collections); each thread processes this range
        for i in irange
            @inbounds index = indices[i]
            @inline f(index)
        end
    end
end


"""
    foreachindex(
        f, itr, backend::Backend=get_backend(itr);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Parallelised `for` loop over the indices of an iterable.

It allows you to run normal Julia code on a GPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one GPU thread per index.

On CPUs at most `max_tasks` threads are launched, or fewer such that each thread processes at least
`min_elems` indices; if a single task ends up being needed, `f` is inlined and no thread is
launched. Tune it to your function - the more expensive it is, the fewer elements are needed to
amortise the cost of launching a thread (which is a few μs).

# Examples
Normally you would write a for loop like this:
```julia
function f()
    x = Array(1:100)
    y = similar(x)
    for i in eachindex(x)
        @inbounds y[i] = 2 * x[i] + 1
    end
end
```

Using this function you can have the same for loop body over a GPU array:
```julia
using CUDA
import AcceleratedKernels as AK

function f()
    x = CuArray(1:100)
    y = similar(x)
    AK.foreachindex(x) do i
        @inbounds y[i] = 2 * x[i] + 1
    end
end
```

Note that the above code is pure arithmetic, which you can write directly (and on some platforms
it may be faster) as:
```julia
using CUDA
x = CuArray(1:100)
y = 2 .* x .+ 1
```

**Important note**: to use this function on a GPU, the objects referenced inside the loop body must
have known types - i.e. be inside a function. For example:
```julia
using oneAPI
import AcceleratedKernels as AK

x = oneArray(1:100)

# CRASHES - typical error message: "Reason: unsupported dynamic function invocation"
# AK.foreachindex(x) do i
#     x[i] = i
# end

function somecopy!(v)
    # Because it is inside a function, the type of `v` will be known
    AK.foreachindex(v) do i
        v[i] = i
    end
end

somecopy!(x)    # This works
```
"""
function foreachindex(
    f, itr, backend::Backend=get_backend(itr);

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size=256,
)
    if use_gpu_algorithm(backend, prefer_threads)
        _forindices_gpu(f, eachindex(itr), backend; block_size)
    else
        _forindices_threads(f, eachindex(itr); max_tasks, min_elems)
    end
end


"""
    foraxes(
        f, itr, dims::Union{Nothing, <:Integer}=nothing, backend::Backend=get_backend(itr);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Parallelised `for` loop over the indices along axis `dims` of an iterable.

It allows you to run normal Julia code on a GPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one GPU thread per index.

On CPUs at most `max_tasks` threads are launched, or fewer such that each thread processes at least
`min_elems` indices; if a single task ends up being needed, `f` is inlined and no thread is
launched. Tune it to your function - the more expensive it is, the fewer elements are needed to
amortise the cost of launching a thread (which is a few μs).

# Examples
Normally you would write a for loop like this:
```julia
function f()
    x = Array(reshape(1:30, 3, 10))
    y = similar(x)
    for i in axes(x, 2)
        for j in axes(x, 1)
            @inbounds y[j, i] = 2 * x[j, i] + 1
        end
    end
end
```

Using this function you can have the same for loop body over a GPU array:
```julia
using CUDA
import AcceleratedKernels as AK

function f()
    x = CuArray(reshape(1:3000, 3, 1000))
    y = similar(x)
    AK.foraxes(x, 2) do i
        for j in axes(x, 1)
            @inbounds y[j, i] = 2 * x[j, i] + 1
        end
    end
end
```

**Important note**: to use this function on a GPU, the objects referenced inside the loop body must
have known types - i.e. be inside a function. For example:
```julia
using oneAPI
import AcceleratedKernels as AK

x = oneArray(reshape(1:3000, 3, 1000))

# CRASHES - typical error message: "Reason: unsupported dynamic function invocation"
# AK.foraxes(x) do i
#     x[i] = i
# end

function somecopy!(v)
    # Because it is inside a function, the type of `v` will be known
    AK.foraxes(v) do i
        v[i] = i
    end
end

somecopy!(x)    # This works
```
"""
function foraxes(
    f, itr, dims::Union{Nothing, <:Integer}=nothing, backend::Backend=get_backend(itr);

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size=256,
)
    if isnothing(dims)
        return foreachindex(
            f, itr, backend;
            max_tasks, min_elems,
            prefer_threads, block_size,
        )
    end

    if use_gpu_algorithm(backend, prefer_threads)
        _forindices_gpu(f, axes(itr, dims), backend; block_size)
    else
        _forindices_threads(f, axes(itr, dims); max_tasks, min_elems)
    end
end
