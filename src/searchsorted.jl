# Taken from julia.Base to ensure consistent results with the Base CPU version
# License is MIT: https://julialang.org/license
function _searchsortedfirst(v, x, lo::T, hi::T, comp) where T<:Integer
    hi = hi + T(1)
    len = hi - lo
    @inbounds while len != 0x0
        half_len = len >>> 0x1
        m = lo + half_len
        if comp(v[m], x)
            lo = m + 0x1
            len -= half_len + 0x1
        else
            hi = m
            len = half_len
        end
    end
    return lo
end


function _searchsortedfirst(v, x, lo::T, hi::T, ord::Base.Order.Ordering) where T<:Integer
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


function _searchsortedlast(v, x, lo::T, hi::T, comp) where T<:Integer
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = lo + ((hi - lo) >>> 0x1)
        if comp(x, v[m])
            hi = m
        else
            lo = m
        end
    end
    return lo
end


function _searchsortedlast(v, x, lo::T, hi::T, ord::Base.Order.Ordering) where T<:Integer
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
    backend = _resolve_backend(backend, ix, v, xs)
    length(ix) == length(xs) || throw(ArgumentError(
        "index array must have as many elements as the queries, $(length(ix)) != $(length(xs))"))
    ord = Base.Order.ord(lt, by, rev, order)
    _foreachindex(eachindex(xs), backend; block_size, max_tasks, min_elems) do i
        @inbounds ix[i] = _searchsortedfirst(v, xs[i], firstindex(v), lastindex(v), ord)
    end
    ix
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
    backend = _resolve_backend(backend, ix, v, xs)
    length(ix) == length(xs) || throw(ArgumentError(
        "index array must have as many elements as the queries, $(length(ix)) != $(length(xs))"))
    ord = Base.Order.ord(lt, by, rev, order)
    _foreachindex(eachindex(xs), backend; block_size, max_tasks, min_elems) do i
        @inbounds ix[i] = _searchsortedlast(v, xs[i], firstindex(v), lastindex(v), ord)
    end
    ix
end
