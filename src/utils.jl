# Local memory that kernels with a tunable footprint may use: the minimum an OpenCL device must
# provide (`CL_DEVICE_LOCAL_MEM_SIZE`, full profile), and Metal's threadgroup memory limit.
const LOCAL_MEMORY_BUDGET = 32 * 1024

# The default of every `init` keyword: no initial value given. As in Base, this differs from
# `init=nothing`, which is an explicit initial value.
struct NoInit end

@inline unval(::Val{V}) where {V} = V


# An array for an operation's result: `similar(v, ...)`, unless `v` has no backend (a range, for
# instance), which gives a new array on `backend`
similar_on(backend, v, ::Type{T}=eltype(v), dims=size(v)) where {T} =
    backendless(v) ? KernelAbstractions.allocate(backend, T, dims) : similar(v, T, dims)
# A copy of `v`, likewise
copy_on(backend, v) = backendless(v) ? copyto!(similar_on(backend, v), collect(v)) : copy(v)

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
    has_offset_parent(dst) || has_offset_parent(src) || return copyto!(dst, src)
    foreachindex_on(Base.OneTo(length(dst)), backend) do i
        @inbounds dst[i] = src[i]
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
