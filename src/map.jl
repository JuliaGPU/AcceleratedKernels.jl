"""
    map!(
        f, dst::AbstractArray, src::AbstractArray;
        backend=nothing,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
    ) -> dst

Apply the function `f` to each element of `src` in parallel and store the result in `dst`, which
must have as many elements. `backend` is derived from `dst` and `src`; the other keywords are
those of [`foreachindex`](@ref).

On CPUs, multithreading only improves performance when complex computation hides the memory
latency and the overhead of spawning tasks - that includes more complex functions and less
cache-local array access patterns. For compute-bound tasks, it scales linearly with the number of
threads.

# Examples
```julia
using Metal
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
    f, dst::AbstractArray, src::AbstractArray;
    backend::Union{Nothing, Backend}=nothing,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
)
    backend = _resolve_backend(backend, dst, src)
    length(dst) == length(src) || throw(ArgumentError(
        "destination and source must have the same length, $(length(dst)) != $(length(src))"))
    _foreachindex(eachindex(src), backend; block_size, max_tasks, min_elems) do idx
        dst[idx] = f(src[idx])
    end
    dst
end


"""
    map(f, src::AbstractArray; kwargs...)

Apply the function `f` to each element of `src` and store the results in a copy of `src` (if `f`
changes the `eltype`, allocate `dst` separately and call [`map!`](@ref)). The keywords are those
of [`map!`](@ref).
"""
function map(f, src::AbstractArray; backend::Union{Nothing, Backend}=nothing, kwargs...)
    backend = _resolve_backend(backend, src)
    return map!(f, _similar(backend, src), src; backend, kwargs...)
end
