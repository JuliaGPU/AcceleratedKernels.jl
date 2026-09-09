# Materialize iterators once so validation does not consume the dimensions used by the loop.
function check_reverse_dims(A, dims)
    dims isa Colon && return dims
    applicable(iterate, dims) ||
        throw(ArgumentError("dims must be an integer, an iterable of integers, or `:`, got $dims"))
    dims = Tuple(dims)
    for d in dims
        d isa Integer ||
            throw(ArgumentError("dims must be integers, got $dims"))
        1 <= d <= ndims(A) ||
            throw(ArgumentError("dimension $d is out of range 1:$(ndims(A))"))
    end
    allunique(dims) || throw(ArgumentError("dims $dims contains duplicates"))
    return dims
end


# Preserve the distinction between omitted bounds and an explicitly requested whole range.
function check_reverse_range(v, dims, start::Union{Nothing,Integer}, stop::Union{Nothing,Integer})
    isnothing(start) && isnothing(stop) && return nothing
    dims isa Colon ||
        throw(ArgumentError("`start`/`stop` cannot be combined with `dims`"))
    v isa AbstractVector ||
        throw(ArgumentError("`start`/`stop` are only supported for vectors"))
    start = isnothing(start) ? firstindex(v) : Int(start)
    stop = isnothing(stop) ? lastindex(v) : Int(stop)
    # Base.reverse! skips bounds checks when no pair needs swapping.
    stop > start && checkbounds(v, start:stop)
    return start:stop
end


# Split along the last non-singleton reversed dimension. For odd extents, the middle
# slice mirrors onto itself; the index ordering guard swaps each pair in it only once.
function reverse_dims!(
    v::AbstractArray{T, N}, dims, backend;
    kwargs...
) where {T, N}
    rev_dims = ntuple(d -> (d in dims) && size(v, d) > 1, N)
    half_dim = findlast(rev_dims)
    isnothing(half_dim) && return v

    ref = size(v) .+ 1
    lin_idx = LinearIndices(v)
    reduced_size = ntuple(d -> ifelse(d == half_dim, cld(size(v, d), 2), size(v, d)), N)
    nd_idx = CartesianIndices(reduced_size)

    foreachindex(1:Base.prod(reduced_size), backend; kwargs...) do i
        idx = Tuple(nd_idx[i])
        index_in = lin_idx[idx...]
        idx_mirror = ifelse.(rev_dims, ref .- idx, idx)
        index_out = lin_idx[idx_mirror...]
        @inbounds if index_in < index_out
            temp = v[index_out]
            v[index_out] = v[index_in]
            v[index_in] = temp
        end
    end

    v
end


function reverse_dims!(
    dst::AbstractArray, src::AbstractArray{T, N}, dims, backend;
    kwargs...
) where {T, N}
    rev_dims = ntuple(d -> (d in dims) && size(src, d) > 1, N)
    ref = size(src) .+ 1
    lin_idx = LinearIndices(src)
    nd_idx = CartesianIndices(src)

    foreachindex(src, backend; kwargs...) do i
        idx = Tuple(nd_idx[i])
        idx_mirror = ifelse.(rev_dims, ref .- idx, idx)
        index_out = lin_idx[idx_mirror...]
        @inbounds dst[index_out] = src[i]
    end

    dst
end


"""
    reverse!(
        v::AbstractArray, backend::Backend=get_backend(v);

        dims=:,
        start=nothing,
        stop=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Reverse `v` in-place and return it. With `dims=:` (the default) the whole array is reversed; pass
`dims=d` (an integer or an iterable of distinct integers in `1:ndims(v)`) to reverse only along
those dimensions. `dims=()` leaves `v` unchanged. For a vector, `start`/`stop` restrict the
reversal to the sub-range `v[start:stop]`, as in `Base.reverse!(v, start, stop)`; they cannot be
combined with `dims` other than `:`. Omitted bounds default to `firstindex(v)` and `lastindex(v)`.
When `start >= stop`, nothing is reversed and bounds are not checked, as in Base's in-place
method. Otherwise both bounds must lie within the vector. The CPU and GPU settings are the
same as for [`foreachindex`](@ref).

No temporary array is allocated.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(1:100_000)
AK.reverse!(v)
AK.reverse!(v; start=10, stop=20)  # reverse only v[10:20]

m = CUDA.CuArray(reshape(1:12, 3, 4))
AK.reverse!(m; dims=2)          # reverse the columns
```
"""
function reverse!(
    v::AbstractArray, backend::Backend=get_backend(v);
    dims=:, start=nothing, stop=nothing, kwargs...
)
    dims = check_reverse_dims(v, dims)
    range = check_reverse_range(v, dims, start, stop)
    if !isnothing(range)
        first(range) >= last(range) && return v
        reverse!(view(v, range), backend; kwargs...)
        return v
    end
    if !(dims isa Colon)
        return reverse_dims!(v, dims, backend; kwargs...)
    end

    len = length(v)
    len <= 1 && return v

    lo = firstindex(v)
    hi = lastindex(v)

    # Swap each pair once; an odd-length array keeps its middle element.
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

        dims=:,
        start=nothing,
        stop=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Write the reverse of `src` into `dst` and return `dst`; `src` is left unchanged. `dst` and `src`
must not alias. With `dims=:` (the default), the whole array is reversed and only the lengths
must match. With an integer or iterable of distinct dimensions, the sizes must match. `dims=()`
copies `src` unchanged. For a vector `src`, `start`/`stop` reverse only `src[start:stop]` and copy
the rest of `src` unchanged. Bounds default to `firstindex(src)` and `lastindex(src)` and cannot
be combined with `dims` other than `:`. When `start >= stop`, this copies `src` without checking
the bounds, following the in-place method; Base's allocating `reverse` may throw for these
out-of-bounds ranges. Otherwise both bounds must lie within `src`.
Elements are converted to the destination element type on assignment.
The CPU and GPU settings are the same as for [`foreachindex`](@ref).
"""
function reverse!(
    dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);
    dims=:, start=nothing, stop=nothing, kwargs...
)
    dims = check_reverse_dims(src, dims)
    range = check_reverse_range(src, dims, start, stop)
    if !isnothing(range)
        @argcheck length(dst) == length(src)
        isempty(src) && return dst
        # Fuse copying and reversal to avoid a second pass and synchronizing device copies.
        lo, hi = first(range), last(range)
        shift = firstindex(dst) - firstindex(src)
        foreachindex(src, backend; kwargs...) do i
            j = lo <= i <= hi ? lo + hi - i : i
            @inbounds dst[j + shift] = src[i]
        end
        return dst
    end
    if !(dims isa Colon)
        @argcheck size(dst) == size(src)
        length(src) == 0 && return dst
        return reverse_dims!(dst, src, dims, backend; kwargs...)
    end

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

        dims=:,
        start=nothing,
        stop=nothing,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Return a reversed copy of `v`, leaving `v` unchanged. With `dims=:` (the default) the whole array is
reversed; pass `dims=d` to reverse only along those dimensions, or, for a vector, `start`/`stop` to
reverse only the sub-range `v[start:stop]`. Bounds and their defaults follow [`reverse!`](@ref),
including copying unchanged when `start >= stop` even if the bounds lie outside the vector.
The CPU and GPU settings are the same as for [`foreachindex`](@ref).

Prefer [`reverse!`](@ref) when you do not need to keep `v`; it avoids the allocation.
"""
function reverse(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    reverse!(similar(v), v, backend; kwargs...)
end
