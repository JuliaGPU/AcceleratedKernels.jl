# Copied from Base, so that the results match `Base.searchsortedfirst` and `searchsortedlast`
# License is MIT: https://julialang.org/license
function searchsorted_first(v, x, lo::T, hi::T, ord::Base.Order.Ordering) where T<:Integer
    hi = hi + T(1)
    len = hi - lo
    @inbounds while len != 0x0
        half_len = len >>> 0x1
        m = lo + half_len
        if Base.Order.lt(ord, v[m], x)
            lo = m + 0x1
            len -= half_len + 0x1
        else
            hi = m
            len = half_len
        end
    end
    return lo
end


function searchsorted_last(v, x, lo::T, hi::T, ord::Base.Order.Ordering) where T<:Integer
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = lo + ((hi - lo) >>> 0x1)
        if Base.Order.lt(ord, x, v[m])
            hi = m
        else
            lo = m
        end
    end
    return lo
end


# C++'s `std::lower_bound` and `std::upper_bound` with zero-based positions, for the merge sort
# kernels, whose index arithmetic is zero-based: the first position in `[left, right)` whose
# element is not less than `value` (`lower_bound0`) or is greater than `value` (`upper_bound0`)
# under the less-than `comp`, or `right` if there is none.
Base.@propagate_inbounds function lower_bound0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(arr[right], value) && return right
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(arr[mid], value)
            left = mid
        else
            right = mid
        end
    end
    return left
end


Base.@propagate_inbounds function upper_bound0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(value, arr[left + 0x1]) && return left
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(value, arr[mid + 0x1])
            right = mid
        else
            left = mid
        end
    end
    return right
end


"""
    searchsortedfirst!(
        ix::AbstractVector, v::AbstractVector, xs::AbstractVector;
        backend=nothing,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1000,
    ) -> ix

Batched binary search: for each element `xs[i]`, write `searchsortedfirst(v, xs[i]; lt, by, rev,
order)` into `ix[i]`, the index of the first element of the sorted vector `v` not ordered before
`xs[i]`. Unlike `Base.searchsortedfirst`, which would treat `xs` as a single value, every element
of `xs` is a separate query. `ix` needs as many elements as `xs`.

`backend` is derived from `ix`, `v` and `xs`; the launch keywords are those of
[`foreachindex`](@ref), except that on the host at least `min_elems=1000` queries go to each
task.

# Examples
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(sort(rand(Float32, 1_000_000)))
xs = CuArray(rand(Float32, 10_000))
ix = similar(xs, Int)
AK.searchsortedfirst!(ix, v, xs)
```
"""
function searchsortedfirst!(
    ix::AbstractVector, v::AbstractVector, xs::AbstractVector;
    backend::Union{Nothing, Backend}=nothing,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,
)
    searchsorted_each!(searchsorted_first, ix, v, xs, resolve_backend(backend, ix, v, xs),
                       Base.Order.ord(lt, by, rev, order); block_size, max_tasks, min_elems)
end


"""
    searchsortedlast!(
        ix::AbstractVector, v::AbstractVector, xs::AbstractVector;
        backend=nothing,
        lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,
        block_size::Int=256,
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1000,
    ) -> ix

Batched binary search: for each element `xs[i]`, write `searchsortedlast(v, xs[i]; lt, by, rev,
order)` into `ix[i]`, the index of the last element of the sorted vector `v` not ordered after
`xs[i]`. The keywords are those of [`searchsortedfirst!`](@ref).
"""
function searchsortedlast!(
    ix::AbstractVector, v::AbstractVector, xs::AbstractVector;
    backend::Union{Nothing, Backend}=nothing,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    block_size::Int=256,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1000,
)
    searchsorted_each!(searchsorted_last, ix, v, xs, resolve_backend(backend, ix, v, xs),
                       Base.Order.ord(lt, by, rev, order); block_size, max_tasks, min_elems)
end

# `ix[i] = search(v, xs[i], ...)` for every query
function searchsorted_each!(search, ix, v, xs, backend, ord; launch...)
    length(ix) == length(xs) || throw(ArgumentError(
        "index array must have as many elements as the queries, $(length(ix)) != $(length(xs))"))
    # (`ix` and `xs` are paired by position)
    ix1, xs1 = one_based(ix), one_based(xs)
    foreachindex_on(eachindex(xs1), backend; launch...) do i
        @inbounds ix1[i] = search(v, xs1[i], firstindex(v), lastindex(v), ord)
    end
    ix
end
