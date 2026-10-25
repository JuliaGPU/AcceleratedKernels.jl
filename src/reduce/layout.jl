# How a reduction along `dims` indexes its source: the reduced and kept dimensions merged into
# segments, and the buffer, offset, strides and vector width of the source's memory.

# Split the dimensions into reduced and kept ("outer") segments, merging adjacent dimensions of
# the same kind where the second's stride continues the first's. Returns the `(stride, size)`
# pairs of the reduced and of the kept segments, each as `N`-tuples of strides and sizes padded
# with zeros, and the number of segments. (The fixed length keeps this type stable: the reduction's
# host path runs on every call, and dynamic dispatch there costs more than small kernels take.)
function merge_dims(src_sizes::NTuple{N, Int}, src_strides::NTuple{N, Int}, rdims) where {N}
    pad = ntuple(_ -> 0, Val(N))
    rstrides, rsizes, nr = pad, pad, 0
    ostrides, osizes, no = pad, pad, 0
    i = 1
    while i <= N
        reduced = i in rdims
        seg_stride = src_strides[i]
        seg_size   = src_sizes[i]
        while i + 1 <= N && ((i + 1) in rdims) == reduced &&
              src_strides[i + 1] == seg_stride * seg_size
            i += 1
            seg_size *= src_sizes[i]
        end
        if reduced
            nr += 1
            rstrides = Base.setindex(rstrides, seg_stride, nr)
            rsizes = Base.setindex(rsizes, seg_size, nr)
        else
            no += 1
            ostrides = Base.setindex(ostrides, seg_stride, no)
            osizes = Base.setindex(osizes, seg_size, no)
        end
        i += 1
    end
    return (rstrides, rsizes, nr), (ostrides, osizes, no)
end

# Call `g(reduce_strides, reduce_sizes, outer_strides, outer_sizes)` with the segments of
# `merge_dims` as tuples of their lengths, which the kernels specialize on. The common
# layouts (one reduced segment and at most one kept one) take no dynamic dispatch.
@inline function with_segments(g::G, segs) where {G}
    (rstrides, rsizes, nr), (ostrides, osizes, no) = segs
    if nr == 1 && no == 1
        return g((rstrides[1],), (rsizes[1],), (ostrides[1],), (osizes[1],))
    elseif nr == 1 && no == 0
        return g((rstrides[1],), (rsizes[1],), (), ())
    else
        return g(firstn(rstrides, nr), firstn(rsizes, nr),
                 firstn(ostrides, no), firstn(osizes, no))
    end
end
firstn(t::Tuple, n::Int) = ntuple(i -> t[i], n)

# The source of a reduction along `dims` into a single output as a linearly indexed collection
# of its elements, for the whole-array kernels (`mapreduce_1d_gpu`): a `Broadcasted` object, an
# array without a dense buffer, or a contiguous range of a dense buffer; `nothing` for other
# sources.
linear_source(src::Base.Broadcast.Broadcasted, rdims) = src
function linear_source(src::AbstractArray, rdims)
    layout = strided_layout(src)
    isnothing(layout) && return src
    (rstrides, _, nr), _ = merge_dims(size(src), layout.strides, rdims)
    nr == 1 && rstrides[1] == 1 || return nothing
    (; buffer, offset) = layout
    return offset == 0 && buffer === src ? src : view(buffer, offset + 1:offset + length(src))
end

# (the strides of a dense column-major array: 1 and the running products of the sizes)
dense_strides(sizes::Tuple{}) = ()
dense_strides(sizes::Tuple) = Base.front(tuple_cumprod((1, sizes...)))
tuple_cumprod(t::Tuple{Any}) = t
tuple_cumprod(t::Tuple) = (t[1], tuple_cumprod((t[1] * t[2], Base.tail(Base.tail(t))...))...)

# The layout the reduction kernels index `src` with, as `(; buffer, offset, strides)`: the
# strided layout of a source backed by a dense buffer, else (for a lazy array or a `Broadcasted`
# object) the source indexed linearly, as a dense array of its size. Indexing a lazy source
# linearly costs a decode of the index per element, but it gets the launch shapes of a dense
# array, which can spread a reduction into few outputs across many threads.
function source_layout(src)
    layout = strided_layout(src)
    return isnothing(layout) ? linear_layout(src) : layout
end
linear_layout(src) = (; buffer=linear_view(src), offset=0, strides=dense_strides(size(src)))
# (a reshaped array decodes a linear index with multiplicative inverses rather than divisions)
linear_view(src::AbstractArray) = vec(src)
linear_view(src::Base.Broadcast.Broadcasted) = src

# The layout of a source backed by a single dense column-major buffer, which the kernels index
# at `buffer[offset + Σ coordᵈ·strideᵈ + 1]`: a dense array, or a strided view, adjoint,
# permuted-dims array or reshape of one. `nothing` for other sources (lazy arrays, `Broadcasted`
# objects, complex adjoints, wrappers of wrappers), which are indexed linearly (see
# `source_layout`).
function strided_layout(src::AbstractArray)
    s = try
        strides(src)
    catch err
        # (an `ArgumentError` for a reshape of a non-contiguous view, on some Julia versions)
        err isa Union{MethodError, ArgumentError} || rethrow()
        return nothing
    end

    s == dense_strides(size(src)) && return (; buffer=src, offset=0, strides=s)

    # (a wrapper of a dense array: its parent, at the source's offset)
    p = parent(src)
    (p === src || !isdense(p)) && return nothing
    offset = parent_offset(src, p)
    offset === nothing && return nothing
    return (; buffer=p, offset, strides=s)
end
strided_layout(src::Base.Broadcast.Broadcasted) = nothing

function isdense(p::AbstractArray)
    try
        return strides(p) == dense_strides(size(p))
    catch err
        # (an `ArgumentError` for a reshape of a non-contiguous view, on some Julia versions)
        err isa Union{MethodError, ArgumentError} || rethrow()
        return false
    end
end

# The offset (in elements, from 0) of the wrapper's first element in its parent: a view's first
# index, and 0 for the other wrappers, which start at the parent's first element
function parent_offset(src::SubArray, p)
    try
        # Base.map (not AK's array `map`, which shadows it inside this module).
        first_index = Base.map(first, parentindices(src))
        return LinearIndices(p)[first_index...] - 1
    catch err
        # (index types without `first`, or without a linear index in the parent)
        err isa Union{MethodError, ArgumentError, BoundsError} || rethrow()
        return nothing
    end
end
parent_offset(src, p) = 0

# (`pointer` and its conversion to an address are not defined for every backend's arrays)
function isaligned16(buffer)
    applicable(pointer, buffer) || return false
    ptr = try
        pointer(buffer)
    catch
        return false
    end
    applicable(UInt, ptr) || return false
    return UInt(ptr) % 16 == 0
end

# The width of the vector loads of the by-block kernel for slices of consecutive elements of a
# dense buffer: 16 bytes of 4- or 8-byte elements, where the slices and the buffer are aligned to
# that; else 0
function vector_width(
    buffer, base_offset, outer_strides, reduce_strides, reduce_size, block_size,
)
    T = eltype(buffer)
    isprimitivetype(T) && sizeof(T) in (4, 8) || return 0

    W = 16 ÷ sizeof(T)
    reduce_strides == (1,) || return 0
    base_offset % W == 0 || return 0
    Base.all(s -> s % W == 0, outer_strides) || return 0
    isaligned16(buffer) || return 0

    # (the kernel's 32-bit indices must not wrap with the last grid-stride step)
    UInt128(reduce_size) + UInt128(block_size * W) <= typemax(UInt32) || return 0
    return W
end

# The indices of a slice relative to its output: the axes of the reduced dimensions, and a single
# index along the kept ones, so that `max(Iother, Ireduce)` indexes the source as Base's
# `mapreducedim!` does
slice_indices(src, dst) =
    CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))
