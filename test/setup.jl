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

# The algorithm of tests that do not choose one: `Auto()`, except where `HOST_KERNELS`, whose point
# is to run AK's kernels on the host backend, on which `Auto()` picks the threaded algorithms.
# Each test file also has a function that builds an algorithm of the configuration from the
# settings of both kinds (`reduce_alg`, ...).
default_alg(kernel_alg) = HOST_KERNELS ? kernel_alg : AK.Auto()

# The element types the backend supports
valid_backend_eltypes(backend, eltypes) = filter(T -> T !== Float64 || KernelAbstractions.supports_float64(backend), eltypes)

# Whether the backend's arrays can hold bits unions such as `Union{Missing, Bool}` (Metal.jl's,
# oneAPI.jl's and OpenCL.jl's cannot yet)
const TEST_BITS_UNIONS = nameof(typeof(BACKEND)) ∉ (:MetalBackend, :oneAPIBackend, :OpenCLBackend)

using Adapt: Adapt, adapt

# An array whose axes start at `1 + offsets[d]`, like an `OffsetArray`
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

# A lazy array, without a dense buffer: its elements are computed from two arrays
struct LazyProduct{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    a::A
    b::A
end
Base.size(p::LazyProduct) = size(p.a)
Base.@propagate_inbounds Base.getindex(p::LazyProduct{T, N}, I::Vararg{Int, N}) where {T, N} =
    p.a[I...] * p.b[I...]
Adapt.adapt_structure(to, p::LazyProduct) = LazyProduct(adapt(to, p.a), adapt(to, p.b))
KernelAbstractions.get_backend(p::LazyProduct) = get_backend(p.a)

# An associative but non-commutative operator: 2x2 matrix products over UInt32, which wraps and so
# forms a ring. Random products of SL(2, Z) generators stay invertible, so products of many of
# them never collapse to zero (as products of arbitrary matrices do, hiding ordering mistakes on
# long inputs).
struct Mat2
    a::UInt32
    b::UInt32
    c::UInt32
    d::UInt32
end
Base.zero(::Type{Mat2}) = Mat2(0, 0, 0, 0)
matmul2(x::Mat2, y::Mat2) = Mat2(
    x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d,
    x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d,
)
const I2 = Mat2(1, 0, 0, 1)
const MAT2_GENERATORS = (Mat2(1, 1, 0, 1), Mat2(1, 0, 1, 1),
                         Mat2(1, typemax(UInt32), 0, 1), Mat2(1, 0, typemax(UInt32), 1))
randmat2() = foldl(matmul2, rand(MAT2_GENERATORS, 3); init=I2)
