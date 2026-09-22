"""
    map!(
        f, dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);

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
    f, dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);
    kwargs...
)
    @argcheck length(dst) == length(src)
    foreachindex(
        src, backend;
        kwargs...
    ) do idx
        dst[idx] = f(src[idx])
    end
    dst
end


"""
    map(
        f, src::AbstractArray, backend::Backend=get_backend(src);

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
    f, src::AbstractArray, backend::Backend=get_backend(src);
    kwargs...
)
    dst = similar(src)
    map!(
        f, dst, src, backend;
        kwargs...
    )
end


# Multi-argument `map` / `map!`.
#
# Mirrors `AK.mapreduce`'s dispatch: a `Backend` is never an `AbstractArray`, so it can only appear
# as the trailing positional argument. That keeps these additions non-breaking - the single-source
# methods above are unchanged, and `map(f, A, B)` unambiguously means two source arrays. `f` is
# applied across the aligned elements of every source array; the arrays must share axes.

function map!(
    f, dst::AbstractArray, src::AbstractArray, src2::AbstractArray, srcs::AbstractArray...;
    kwargs...
)
    _map_multi!(f, dst, nothing, src, src2, srcs...; kwargs...)
end

function map!(
    f, dst::AbstractArray, src::AbstractArray, src2::AbstractArray, arg, args...;
    kwargs...
)
    backend = isempty(args) ? arg : args[end]
    backend isa Backend || throw(MethodError(map!, (f, dst, src, src2, arg, args...)))
    srcs = isempty(args) ? () : (arg, args[1:end - 1]...)
    _map_multi!(f, dst, backend, src, src2, srcs...; kwargs...)
end

function map(
    f, src::AbstractArray, src2::AbstractArray, srcs::AbstractArray...;
    kwargs...
)
    _map_multi(f, nothing, src, src2, srcs...; kwargs...)
end

function map(
    f, src::AbstractArray, src2::AbstractArray, arg, args...;
    kwargs...
)
    backend = isempty(args) ? arg : args[end]
    backend isa Backend || throw(MethodError(map, (f, src, src2, arg, args...)))
    srcs = isempty(args) ? () : (arg, args[1:end - 1]...)
    _map_multi(f, backend, src, src2, srcs...; kwargs...)
end

# Index every source array at `idx`, returning the aligned elements as a tuple to splat into `f`.
# A named, type-annotated helper keeps `idx` an argument rather than a capture of a closure nested
# inside the kernel closure, which is what lets it compile on every backend.
@inline _map_indextuple(sources::Tuple, idx)::Tuple = Base.map(s -> @inbounds(s[idx]), sources)

function _map_check_axes(src::AbstractArray, srcs::AbstractArray...)
    src_axes = axes(src)
    for other in srcs
        axes(other) == src_axes ||
            throw(DimensionMismatch("all arrays in `map`/`map!` must have the same axes"))
    end
    return nothing
end

function _map_multi!(
    f, dst::AbstractArray, backend::Union{Nothing, Backend},
    src::AbstractArray, srcs::AbstractArray...;
    kwargs...
)
    _map_check_axes(dst, src, srcs...)
    be = isnothing(backend) ? get_backend(src) : backend
    sources = (src, srcs...)
    # Apply `f` across the aligned elements by indexing each source array directly and splatting the
    # results. Indexing the arrays themselves (rather than a fused `Broadcasted`) means a single
    # linear index works for any dimensionality on every supported Julia version, and there is no
    # intermediate to materialize.
    foreachindex(dst, be; kwargs...) do idx
        @inbounds dst[idx] = f(_map_indextuple(sources, idx)...)
    end
    dst
end

function _map_multi(
    f, backend::Union{Nothing, Backend},
    src::AbstractArray, srcs::AbstractArray...;
    kwargs...
)
    _map_check_axes(src, srcs...)
    dst = similar(src, Base.Broadcast.combine_eltypes(f, (src, srcs...)))
    _map_multi!(f, dst, backend, src, srcs...; kwargs...)
end
