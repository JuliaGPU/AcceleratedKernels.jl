# The helpers of the generic tests, included after the globals runtests.jl sets for the backend

# A copy of the host array on the backend (on the host for the threaded algorithms)
array_from_host(h_arr::AbstractArray, dtype=nothing) = array_from_host(BACKEND, h_arr, dtype)
function array_from_host(backend, h_arr::AbstractArray, dtype=nothing)
    d_arr = if !TEST_KERNELS
        zeros(isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
    else
        KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
    end
    copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))
    d_arr
end

# Call `op` with the workspace made for the same call
with_workspace(op, args...; kwargs...) =
    op(args...; kwargs..., workspace=AK.workspace(op, args...; kwargs...))

# The element types the backend supports
valid_backend_eltypes(backend, eltypes) = filter(T -> T !== Float64 || KernelAbstractions.supports_float64(backend), eltypes)

# Whether the backend's arrays can hold bits unions such as `Union{Missing, Bool}` (Metal.jl's,
# oneAPI.jl's and OpenCL.jl's cannot yet)
const TEST_BITS_UNIONS = nameof(typeof(BACKEND)) ∉ (:MetalBackend, :oneAPIBackend, :OpenCLBackend)

# An array whose axes start at `1 + offsets[d]`, like an `OffsetArray`
using Adapt: Adapt, adapt
struct OffsetAxes{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
    offsets::NTuple{N, Int}
end
Base.axes(x::OffsetAxes) =
    Base.map((ax, o) -> Base.IdentityUnitRange(ax .+ o), axes(x.parent), x.offsets)
Base.size(x::OffsetAxes) = size(x.parent)
Base.@propagate_inbounds Base.getindex(x::OffsetAxes{T, N}, I::Vararg{Int, N}) where {T, N} =
    x.parent[(I .- x.offsets)...]
Base.@propagate_inbounds Base.setindex!(x::OffsetAxes{T, N}, v, I::Vararg{Int, N}) where {T, N} =
    x.parent[(I .- x.offsets)...] = v
Base.similar(x::OffsetAxes, ::Type{T}, dims::Dims) where {T} = similar(x.parent, T, dims)
Base.copy(x::OffsetAxes) = OffsetAxes(copy(x.parent), x.offsets)
Adapt.adapt_structure(to, x::OffsetAxes) = OffsetAxes(adapt(to, x.parent), x.offsets)
KernelAbstractions.get_backend(x::OffsetAxes) = get_backend(x.parent)

offset_from_host(h_arr::AbstractArray, offsets) = OffsetAxes(array_from_host(h_arr), offsets)
