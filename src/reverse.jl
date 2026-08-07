"""
    reverse!(
        v::AbstractArray, backend::Backend=get_backend(v);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Reverse `v` in-place and return it. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).

Each thread swaps one symmetric pair `v[i] <-> v[end - i + 1]`, so only `length(v) ÷ 2` threads
are launched and no temporary array is allocated. Arrays of odd length keep their middle element
in place.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(1:100_000)
AK.reverse!(v)
```
"""
function reverse!(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    len = length(v)
    len <= 1 && return v

    lo = firstindex(v)
    hi = lastindex(v)

    # Only the lower half needs threads - each one swaps its mirrored partner too; for odd
    # lengths the middle element is its own mirror, so it is correctly left untouched
    foreachindex(1:(len ÷ 2), backend; kwargs...) do i
        left = lo + i - 1
        right = hi - i + 1
        @inbounds begin
            temp = v[left]
            v[left] = v[right]
            v[right] = temp
        end
    end

    v
end


"""
    reverse!(
        dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Write the reverse of `src` into `dst` and return `dst`; `src` is left unchanged. `dst` and `src`
must have the same length and must not alias. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).
"""
function reverse!(
    dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);
    kwargs...
)
    @argcheck length(dst) == length(src)
    length(src) == 0 && return dst

    hi_src = lastindex(src)
    lo_dst = firstindex(dst)

    foreachindex(src, backend; kwargs...) do i
        @inbounds dst[lo_dst + (hi_src - i)] = src[i]
    end

    dst
end


"""
    reverse(
        v::AbstractArray, backend::Backend=get_backend(v);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Return a reversed copy of `v`, leaving `v` unchanged. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).

Prefer [`reverse!`](@ref) when you do not need to keep `v` - it avoids the allocation.
"""
function reverse(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    dst = similar(v)
    reverse!(dst, v, backend; kwargs...)
end
