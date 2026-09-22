"""
    map!(
        f, dst::AbstractArray, src::AbstractArray, srcs::AbstractArray...;
        backend=nothing,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
    ) -> dst

Apply the function `f` to each element of `src` in parallel and store the result in `dst`. With
more source arrays, `f` takes one element of each, as in `Base.map!`. `dst` and the sources must
have the same number of elements, which are matched in column-major order, so their shapes may
differ. `dst` may be one of the sources, but must not otherwise share memory with them (a shifted
view of a source, say). `backend` is derived from `dst` and the sources; the other keywords are
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

z = similar(x)
AK.map!(+, z, x, y)
```
"""
function map!(
    f, dst::AbstractArray, src::AbstractArray, srcs::AbstractArray...;
    backend::Union{Nothing, Backend}=nothing,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
)
    srcs = (src, srcs...)
    backend = _resolve_backend(backend, dst, srcs...)
    for s in srcs
        length(s) == length(dst) || throw(ArgumentError(
            "destination and sources must have the same length, $(length(dst)) != $(length(s))"))
    end
    launch = (; block_size, max_tasks, min_elems)
    if Base.all(s -> axes(s) == axes(dst), srcs)
        # Index all arrays alike, linearly if they all support it
        _foreachindex(eachindex(dst, srcs...), backend; launch...) do i
            @inbounds dst[i] = f(_getindices(srcs, i)...)
        end
    else
        # Match the elements by their position in column-major order
        _foreachindex(Base.OneTo(length(dst)), backend; launch...) do k
            @inbounds dst[firstindex(dst) - 1 + k] = f(_getnth(srcs, k)...)
        end
    end
    dst
end

# The elements of `arrays` at index `i`, and at position `k` in column-major order
@inline _getindices(arrays::Tuple, i) = Base.map(a -> @inbounds(a[i]), arrays)
@inline _getnth(arrays::Tuple, k) = Base.map(a -> _nth(a, k), arrays)


"""
    map(f, src::AbstractArray, srcs::AbstractArray...; kwargs...)

Apply the function `f` as [`map!`](@ref) does, storing the results in a new array with the shape
and `eltype` of `src` (if `f` changes the `eltype`, allocate `dst` separately and call
[`map!`](@ref)). The keywords are those of [`map!`](@ref).
"""
function map(f, src::AbstractArray, srcs::AbstractArray...;
             backend::Union{Nothing, Backend}=nothing, kwargs...)
    backend = _resolve_backend(backend, src, srcs...)
    return map!(f, _similar(backend, src), src, srcs...; backend, kwargs...)
end
