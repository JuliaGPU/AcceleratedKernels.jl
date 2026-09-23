@kernel inbounds=true cpu=false unsafe_indices=true function _forindices_global!(f, indices)

    # Calculate global index
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N

    # Avoid an out-of-line device call that can copy captured state to local memory.
    if i <= length(indices)
        @inline f(indices[i])
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


# Launch settings of the wrappers, checked on every backend (only one of them applies to each)
function _check_launch(; block_size, max_tasks, min_elems)
    block_size >= 1 || throw(ArgumentError("`block_size` must be positive, got $block_size"))
    max_tasks >= 1 || throw(ArgumentError("`max_tasks` must be positive, got $max_tasks"))
    min_elems >= 1 || throw(ArgumentError("`min_elems` must be positive, got $min_elems"))
    nothing
end

# Run `f` over `indices`: on Julia threads on the host backend, as a kernel elsewhere
function _foreachindex(f, indices, backend::Backend;
                       block_size=256, max_tasks=Threads.nthreads(), min_elems=1)
    _check_launch(; block_size, max_tasks, min_elems)
    if _runs_threads(backend)
        _forindices_threads(f, indices; max_tasks, min_elems)
    else
        _forindices_gpu(f, indices, backend; block_size)
    end
    nothing
end


"""
    foreachindex(
        f, itr;
        backend=nothing,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
    )

Parallelised `for` loop over the indices of an iterable: call `f(i)` for every `i` in
`eachindex(itr)`.

It allows you to run normal Julia code on a GPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one GPU thread per index, in blocks of `block_size` threads.

On the host backend, the loop runs on Julia threads: at most `max_tasks` tasks, or fewer such that
each task processes at least `min_elems` indices; if a single task ends up being needed, `f` is
inlined and no task is launched. Tune it to your function - the more expensive it is, the fewer
elements are needed to amortise the cost of launching a task (which is a few μs).

`backend` is derived from `itr`. A range or other index collection does not determine a backend,
so loops over one run on the host unless you pass `backend`, e.g. the backend of the arrays that
`f` accesses.

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

A loop over a range needs the backend of the arrays it accesses:
```julia
function g!(y, x)
    AK.foreachindex(1:length(x) ÷ 2; backend=AK.get_backend(x)) do i
        @inbounds y[i] = x[2i]
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
    f, itr;
    backend::Union{Nothing, Backend}=nothing,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
)
    backend = _resolve_backend(backend, itr)
    _foreachindex(f, eachindex(itr), backend; block_size, max_tasks, min_elems)
end


"""
    foraxes(
        f, itr, dims::Union{Nothing, Integer}=nothing;
        backend=nothing,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
    )

Parallelised `for` loop over the indices along axis `dims` of an iterable: call `f(i)` for every
`i` in `axes(itr, dims)`, or in `eachindex(itr)` for `dims=nothing`. `dims` mirrors
`axes(itr, dims)`. The keywords are those of [`foreachindex`](@ref).

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

As with [`foreachindex`](@ref), the objects referenced inside the loop body must have known types
on a GPU, i.e. be inside a function.
"""
function foraxes(
    f, itr, dims::Union{Nothing, Integer}=nothing;
    backend::Union{Nothing, Backend}=nothing,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
)
    backend = _resolve_backend(backend, itr)
    indices = isnothing(dims) ? eachindex(itr) : axes(itr, dims)
    _foreachindex(f, indices, backend; block_size, max_tasks, min_elems)
end
