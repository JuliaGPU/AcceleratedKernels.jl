# Reversing. Two regimes:
#   * `dims=:` (default) reverses the whole array, a reversal of the column-major linear order,
#     taken by the fast flat path below (swap mirrored pairs, no index arithmetic).
#   * `dims=d` reverses only along those dimensions, via the general ND kernel below.


# Validate `dims`: a `Colon`, an integer, or an iterable of integers within `1:ndims(A)`.
function _check_reverse_dims(A, dims)
    dims isa Colon && return
    applicable(iterate, dims) || throw(ArgumentError("dimension $dims is not iterable"))
    for d in dims                                       # an integer iterates once
        d isa Integer ||
            throw(ArgumentError("reversed dimension(s) must be integers, got $dims"))
        1 <= d <= ndims(A) ||
            throw(ArgumentError("dimension $dims is not 1 ≤ dims ≤ $(ndims(A))"))
    end
    return
end


# `start`/`stop` select a linear sub-range to reverse; they are only meaningful for the flat
# (`dims=:`) path, so reject the combination with `dims`.
_check_reverse_no_range(v, start, stop) =
    (start == firstindex(v) && stop == lastindex(v)) ||
        throw(ArgumentError("`start`/`stop` cannot be combined with `dims`"))

# A `start`/`stop` sub-range is only defined for vectors; the whole-array default range is allowed.
_check_reverse_vector(v, start, stop) =
    (v isa AbstractVector || (start == firstindex(v) && stop == lastindex(v))) ||
        throw(ArgumentError("`start`/`stop` are only supported for vectors"))

# An empty sub-range (`start > stop`) is a no-op; otherwise both ends must be in bounds.
function _check_reverse_range(v, start, stop)
    start > stop && return
    (firstindex(v) <= start && stop <= lastindex(v)) || throw(BoundsError(v, start:stop))
    return
end


# In-place: split along the last non-singleton reversed dim so only ~half the elements need a
# thread, each swapping with its mirror.
function _reverse_dims!(
    v::AbstractArray{T, N}, dims, backend;
    kwargs...
) where {T, N}
    rev_dims = ntuple(d -> (d in dims) && size(v, d) > 1, N)
    half_dim = findlast(rev_dims)
    isnothing(half_dim) && return v         # all reversed dims are singletons

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


# Out-of-place: one thread per element copies `src[i]` to its mirror slot.
function _reverse_dims!(
    dst::AbstractArray{T, N}, src::AbstractArray{T, N}, dims, backend;
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
        start=firstindex(v),
        stop=lastindex(v),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Reverse `v` in-place and return it. With `dims=:` (the default) the whole array is reversed; pass
`dims=d` (an integer or an iterable of integers) to reverse only along those dimensions, matching
`Base.reverse!`. Alternatively, for a vector, pass `start`/`stop` to reverse only the sub-range
`v[start:stop]`, matching `Base.reverse!(v, start, stop)`; `start`/`stop` are only supported for
vectors and are mutually exclusive with `dims`. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).

For the whole-array case each thread swaps one symmetric pair `v[i] <-> v[end - i + 1]`, so only
`length(v) ÷ 2` threads are launched and no temporary array is allocated. Arrays of odd length keep
their middle element in place.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(1:100_000)
AK.reverse!(v)

AK.reverse!(v; start=3, stop=8)   # reverse only v[3:8] in place

m = CUDA.CuArray(reshape(1:12, 3, 4))
AK.reverse!(m; dims=2)            # reverse the columns
```
"""
function reverse!(
    v::AbstractArray, backend::Backend=get_backend(v);
    dims=:, start::Integer=firstindex(v), stop::Integer=lastindex(v), kwargs...
)
    _check_reverse_dims(v, dims)
    if !(dims isa Colon)
        _check_reverse_no_range(v, start, stop)
        return _reverse_dims!(v, dims, backend; kwargs...)
    end

    _check_reverse_vector(v, start, stop)
    _check_reverse_range(v, start, stop)
    n = stop - start + 1
    n <= 1 && return v

    # Only the lower half needs threads, each swapping its mirrored partner too; for odd
    # lengths the middle element is its own mirror, so it is correctly left untouched
    foreachindex(1:(n ÷ 2), backend; kwargs...) do i
        left = start + i - 1
        right = stop - i + 1
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
        start=firstindex(src),
        stop=lastindex(src),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Write the reverse of `src` into `dst` and return `dst`; `src` is left unchanged. `dst` and `src`
must have the same size and must not alias. With `dims=:` (the default) the whole array is reversed;
pass `dims=d` to reverse only along those dimensions. Alternatively, for a vector, pass `start`/`stop`
to reverse only the sub-range `src[start:stop]`, copying the rest of `src` into `dst` verbatim,
matching `Base.reverse(src, start, stop)`; `start`/`stop` are only supported for vectors and are
mutually exclusive with `dims`. The CPU and GPU settings are the same as for [`foreachindex`](@ref).
"""
function reverse!(
    dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);
    dims=:, start::Integer=firstindex(src), stop::Integer=lastindex(src), kwargs...
)
    _check_reverse_dims(src, dims)
    if !(dims isa Colon)
        _check_reverse_no_range(src, start, stop)
        @argcheck size(dst) == size(src)
        length(src) == 0 && return dst
        return _reverse_dims!(dst, src, dims, backend; kwargs...)
    end

    @argcheck length(dst) == length(src)
    _check_reverse_vector(src, start, stop)
    _check_reverse_range(src, start, stop)

    # Elements outside `[start, stop]` are copied verbatim; only that sub-range is reversed
    len = length(src)
    start > 1   && copyto!(dst, 1, src, 1, start - 1)
    stop  < len && copyto!(dst, stop + 1, src, stop + 1, len - stop)

    n = stop - start + 1
    if n >= 1
        foreachindex(1:n, backend; kwargs...) do k
            @inbounds dst[start + n - k] = src[start + k - 1]
        end
    end

    dst
end


"""
    reverse(
        v::AbstractArray, backend::Backend=get_backend(v);

        dims=:,
        start=firstindex(v),
        stop=lastindex(v),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Return a reversed copy of `v`, leaving `v` unchanged. With `dims=:` (the default) the whole array is
reversed; pass `dims=d` to reverse only along those dimensions, matching `Base.reverse`.
Alternatively, for a vector, pass `start`/`stop` to reverse only the sub-range `v[start:stop]` and
copy the rest verbatim, matching `Base.reverse(v, start, stop)`; `start`/`stop` are only supported
for vectors and are mutually exclusive with `dims`. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).

Prefer [`reverse!`](@ref) when you do not need to keep `v`; it avoids the allocation.
"""
function reverse(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    reverse!(similar(v), v, backend; kwargs...)
end
