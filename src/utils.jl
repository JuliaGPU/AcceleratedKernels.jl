# Local memory that kernels with a tunable footprint may use: the minimum an OpenCL device must
# provide (`CL_DEVICE_LOCAL_MEM_SIZE`, full profile), and Metal's threadgroup memory limit.
const LOCAL_MEMORY_BUDGET = 32 * 1024

# Bytes of local memory per element of type `T` (an isbits union adds a type tag per element)
local_sizeof(::Type{T}) where {T} = Base.aligned_sizeof(T) + Base.isbitsunion(T)

# The default of every `init` keyword: no initial value given. As in Base, this differs from
# `init=nothing`, which is an explicit initial value.
struct NoInit end

@inline unval(::Val{V}) where {V} = V


# The array that holds the memory behind `v`: `v` itself unless it wraps others, else the first of
# the arrays it wraps that holds memory; `nothing` if there is none (a range, or a lazy array over
# one)
storage(v::Tuple) = storage_first(v...)
function storage(v::AbstractArray)
    backendless(v) && return nothing
    p = parent(v)
    p === v ? v : storage(p)
end
storage(_) = nothing
storage_first() = nothing
storage_first(x, xs...) = (s = storage(x); s === nothing ? storage_first(xs...) : s)

# Whether `v` gets Base's `similar` fallback, a host `Array`, although it wraps the memory `s` of
# another array type (a lazy array without a `similar` method of its own, for instance). Checked on
# an empty result.
similar_falls_back(v, s) =
    s !== v && !(s isa Array) && similar(v, eltype(v), Base.map(zero, size(v))) isa Array

# Whether results of operations on `v` are allocated on `backend` rather than with `similar(v)`:
# where `v` holds no memory (a range, or a lazy array over one), or gets Base's fallback
function allocates_on_backend(v)
    s = storage(v)
    s === nothing || similar_falls_back(v, s)
end

# An array for an operation's result
similar_on(backend, v, ::Type{T}=eltype(v), dims=size(v)) where {T} =
    allocates_on_backend(v) ? KernelAbstractions.allocate(backend, T, dims) : similar(v, T, dims)
# A copy of `v`, likewise: its own `copy`, unless its results are allocated on `backend` (where
# Base's copy would give a host `Array`)
function copy_on(backend, v)
    allocates_on_backend(v) && copies_by_kernel(backend, v) &&
        return copyto_on!(similar_on(backend, v), v, backend)
    backendless(v) ? copyto!(similar_on(backend, v), collect(v)) : copy(v)
end
# A copy of `v` in host memory
host_array(backend, v) = copies_by_kernel(backend, v) ?
    Array(copyto_on!(KernelAbstractions.allocate(backend, eltype(v), size(v)), v, backend)) :
    Array(v)

# Whether copies of `v` on `backend` use a kernel. Its own arrays, Base's wrappers of arrays in
# host memory and GPUArrays' wrappers of its arrays are copied by Base or GPUArrays, backendless
# arrays after collecting them on the host, and everything on the host backend. Base's copies of
# other wrappers index device memory element by element on the host, and lazy arrays over ranges
# may read device arrays their function captures.
function copies_by_kernel(backend, v)
    (runs_threads(backend) || backendless(v) || parent(v) === v || v isa AnyGPUArray) &&
        return false
    !(storage(v) isa Union{Array, BitArray})
end

# The destination of an operation (`what`, for the messages) that writes the elements of `src`
# to `dst`: `src` itself where the operation runs in place, else an array that does not overlap
# it, with the same `shape` (`axes`, `size` or `length`)
function check_dst(dst, src, what; shape=axes, in_place::Bool=true)
    in_place && dst === src && return nothing
    Base.mightalias(dst, src) && throw(ArgumentError(in_place ?
        "the destination of $what must be its source or not overlap it" :
        "the destination of $what must not overlap its source"))
    shape(dst) == shape(src) || throw(DimensionMismatch(
        "the destination of $what must have the source's $shape $(shape(src)), got $(shape(dst))"))
    nothing
end

# `A` with axes starting at 1, as the kernels and threaded loops index it, for an `A` whose axes
# start elsewhere (an `OffsetArray`, for instance), and likewise the arrays of a `Broadcasted`
# object. The operations pass these on, and keep `A` for their checks.
one_based(A::AbstractArray) = Base.has_offset_axes(A) ? one_based_view(A) : A
function one_based(bc::Base.Broadcast.Broadcasted)
    Base.has_offset_axes(bc) || return bc
    args = Base.map(one_based, bc.args)
    # (keeping the shape of explicit axes, which may be larger than the arguments')
    return Base.Broadcast.Broadcasted(bc.f, args, Base.map(ax -> Base.OneTo(length(ax)), axes(bc)))
end
one_based(x) = x
# (a reshape keeps the linear indexing of an array of several dimensions, which starts at 1, and
# `OffsetArrays` makes it the parent array; Base does not reshape vectors with offset axes)
one_based_view(A::AbstractVector) = @view A[begin:end]
one_based_view(A::AbstractArray) = reshape(A, size(A))

# Whether `A` is a view `one_based` made. Base's `copyto!` and `fill!` index those element by
# element on the host, which GPU arrays disallow, so `copyto_on!` and `fill_on!` use a kernel.
has_offset_parent(A) =
    A isa Union{SubArray, Base.ReshapedArray} && Base.has_offset_axes(parent(A))

# `copyto!(dst, src)` of arrays of the same size
function copyto_on!(dst, src, backend)
    has_offset_parent(dst) || has_offset_parent(src) || copies_by_kernel(backend, src) ||
        return copyto!(dst, backendless(src) ? collect(src) : src)
    d0, s0 = firstindex(dst), firstindex(src)
    foreachindex_on(Base.OneTo(length(dst)), backend) do i
        @inbounds dst[d0 + i - 1] = src[s0 + i - 1]
    end
    return dst
end

function fill_on!(A, x, backend)
    has_offset_parent(A) || return fill!(A, x)
    foreachindex_on(Base.OneTo(length(A)), backend) do i
        @inbounds A[i] = x
    end
    return A
end
