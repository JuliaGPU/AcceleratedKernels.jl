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

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Reverse `v` in-place and return it. With `dims=:` (the default) the whole array is reversed; pass
`dims=d` (an integer or an iterable of integers) to reverse only along those dimensions, matching
`Base.reverse!`. The CPU and GPU settings are the same as for [`foreachindex`](@ref).

For the whole-array case each thread swaps one symmetric pair `v[i] <-> v[end - i + 1]`, so only
`length(v) ÷ 2` threads are launched and no temporary array is allocated. Arrays of odd length keep
their middle element in place.

To reverse a contiguous sub-range of a vector, reverse a view: `AK.reverse!(@view v[lo:hi])`.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(1:100_000)
AK.reverse!(v)

m = CUDA.CuArray(reshape(1:12, 3, 4))
AK.reverse!(m; dims=2)          # reverse the columns
```
"""
function reverse!(
    v::AbstractArray, backend::Backend=get_backend(v);
    dims=:, kwargs...
)
    _check_reverse_dims(v, dims)
    if !(dims isa Colon)
        return _reverse_dims!(v, dims, backend; kwargs...)
    end

    len = length(v)
    len <= 1 && return v

    lo = firstindex(v)
    hi = lastindex(v)

    # Only the lower half needs threads, each swapping its mirrored partner too; for odd
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

        dims=:,

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Write the reverse of `src` into `dst` and return `dst`; `src` is left unchanged. `dst` and `src`
must have the same size and must not alias. With `dims=:` (the default) the whole array is reversed;
pass `dims=d` to reverse only along those dimensions. The CPU and GPU settings are the same as for
[`foreachindex`](@ref).
"""
function reverse!(
    dst::AbstractArray, src::AbstractArray, backend::Backend=get_backend(src);
    dims=:, kwargs...
)
    _check_reverse_dims(src, dims)
    if !(dims isa Colon)
        @argcheck size(dst) == size(src)
        length(src) == 0 && return dst
        return _reverse_dims!(dst, src, dims, backend; kwargs...)
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

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size=256,
    )

Return a reversed copy of `v`, leaving `v` unchanged. With `dims=:` (the default) the whole array is
reversed; pass `dims=d` to reverse only along those dimensions, matching `Base.reverse`. The CPU and
GPU settings are the same as for [`foreachindex`](@ref).

Prefer [`reverse!`](@ref) when you do not need to keep `v`; it avoids the allocation.
"""
function reverse(
    v::AbstractArray, backend::Backend=get_backend(v);
    kwargs...
)
    reverse!(similar(v), v, backend; kwargs...)
end
