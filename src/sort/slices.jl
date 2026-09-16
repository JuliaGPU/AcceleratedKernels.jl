# Sorting along a dimension works on the 1D slices of an array: for `dims=d`, slice `i` (zero-based)
# holds `len = size(v, d)` elements spaced `stride = prod(size(v)[1:d-1])` apart. The kernels only
# see slices through `slice(v, layout, i)`; the whole-array sort is a flat layout with one slice,
# for which `slice` returns the array itself, keeping its indexing untouched.
struct FlatLayout
    len::Int
end

struct SliceLayout
    len::Int        # elements per slice
    stride::Int     # distance between consecutive elements of a slice
    count::Int      # number of slices
end

slice_layout(v::AbstractArray, ::Colon) = FlatLayout(length(v))

function slice_layout(v::AbstractArray, dims::Integer)
    1 <= dims <= ndims(v) || throw(ArgumentError("dimension out of range"))
    Base.require_one_based_indexing(v)
    len = size(v, dims)
    stride = 1
    for d in 1:dims - 1
        stride *= size(v, d)
    end
    SliceLayout(len, stride, len == 0 ? 0 : length(v) ÷ len)
end

slice_count(::FlatLayout) = 1
slice_count(layout::SliceLayout) = layout.count


# One slice of `parent`, indexed like a vector (one-based)
struct SliceView{A <: AbstractArray}
    parent::A
    offset::Int     # zero-based linear offset of the first element
    stride::Int
end

Base.@propagate_inbounds Base.getindex(s::SliceView, i) =
    s.parent[s.offset + (i - 1) * s.stride + 1]
Base.@propagate_inbounds Base.setindex!(s::SliceView, x, i) =
    s.parent[s.offset + (i - 1) * s.stride + 1] = x
Base.eltype(::Type{SliceView{A}}) where A = eltype(A)

# Split a kernel's linear block index into (slice, block within the slice)
@inline slice_block(::FlatLayout, iblock, blocks_per_slice) = (0, iblock)
@inline slice_block(::SliceLayout, iblock, blocks_per_slice) = divrem(iblock, blocks_per_slice)

@inline slice(v, ::FlatLayout, i) = v

@inline function slice(v, layout::SliceLayout, i)
    # `i` is an index straight from the kernel, so it may be unsigned on some backends
    i = Int(i)
    offset = (i % layout.stride) + (i ÷ layout.stride) * layout.stride * layout.len
    SliceView(v, offset, layout.stride)
end


# CPU: run `f(slice)` on a view of each slice of `v` along `dims`, slices split across tasks
function foreach_slice(f, v::AbstractArray{T, N}, dims::Integer; max_tasks, min_elems) where {T, N}
    1 <= dims <= N || throw(ArgumentError("dimension out of range"))
    others = CartesianIndices(ntuple(d -> d == dims ? Base.OneTo(1) : axes(v, d), N))
    min_slices = cld(min_elems, max(size(v, dims), 1))
    task_partition(length(others), max_tasks, min_slices) do irange
        for i in irange
            other = others[i]
            f(view(v, ntuple(d -> d == dims ? Colon() : other[d], N)...))
        end
    end
    v
end
