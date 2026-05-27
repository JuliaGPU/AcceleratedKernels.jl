"""
    map!(
        f, dst::AbstractArray, src::AbstractArray...;
        backend::Backend=get_backend(src);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Apply the function `f` to each element of `src` in parallel and store the result in `dst`. The
CPU and GPU settings are the same as for [`foreachindex`](@ref).

On CPUs, multithreading only improves performance when complex computation hides the memory
latency and the overhead of spawning tasks - that includes more complex functions and less
cache-local array access patterns. For compute-bound tasks, it scales linearly with the number of
threads.

# Examples
```julia
import Metal
import AcceleratedKernels as AK

x = MtlArray(rand(Float32, 100_000))
y = similar(x)
AK.map!(y, x) do x_elem
    T = typeof(x_elem)
    T(2) * x_elem + T(1)
end
```
"""
function map!(
    f, dst::AbstractArray, src::AbstractArray...;
    backend::Backend=get_backend(src[1]),
    kwargs...
)
    @argcheck lengthcheck(dst, src...)
    foreachindex(
        dst, backend;
        kwargs...
    ) do idx
        dst[idx] = f(indextuple(src, idx)...)
    end
    dst
end


"""
    map(
        f, src::AbstractArray;
        backend::Backend=get_backend(src),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Apply the function `f` to each element of `src` and store the results in a copy of `src` (if `f`
changes the `eltype`, allocate `dst` separately and call [`map!`](@ref)). The CPU and GPU
settings are the same as for [`foreachindex`](@ref).
"""
function map(
    f, src::AbstractArray...;
    backend::Backend=get_backend(src[1]),
    kwargs...
)
    dst = similar(src[1])
    map!(
        f, dst, src...;
        backend, kwargs...
    )
end
