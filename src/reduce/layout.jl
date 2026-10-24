# How a reduction along `dims` indexes its source: the reduced and kept dimensions merged into
# segments, and the buffer, offset, strides and vector width of the source's memory.

# Host-side canonicalization: split the dimensions into reduced and kept ("outer") segments,
# merging adjacent dimensions whose strides are contiguous. Returns the `(stride, size)` pairs of
# the reduced and of the kept segments, each as `N`-tuples of strides and sizes padded with
# zeros, and the number of segments. (The fixed length keeps this type stable: the reduction's
# host path runs on every call, and dynamic dispatch there costs more than small kernels take.)
function _canonicalize_dims(src_sizes::NTuple{N, Int}, src_strides::NTuple{N, Int},
                            dims_valid) where {N}
    pad = ntuple(_ -> 0, Val(N))
    rstrides, rsizes, nr = pad, pad, 0
    ostrides, osizes, no = pad, pad, 0
    i = 1
    while i <= N
        reduced = i in dims_valid
        seg_stride = src_strides[i]
        seg_size   = src_sizes[i]
        while i + 1 <= N && ((i + 1) in dims_valid) == reduced &&
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

# The reduced and kept dimensions of a strided source, as `(strides, sizes, count)` of their
# merged segments (see `_canonicalize_dims`)
_mapreduce_segments(src_sizes, src_strides, dims_valid) =
    _canonicalize_dims(Tuple(src_sizes), Tuple(src_strides), dims_valid)

# Call `g(reduce_strides, reduce_sizes, outer_strides, outer_sizes)` with the segments of
# `_mapreduce_segments` as tuples of their lengths, which the kernels specialize on. The common
# layouts (one reduced segment and at most one kept one) take no dynamic dispatch.
@inline function _with_segments(g::G, segs) where {G}
    (rstrides, rsizes, nr), (ostrides, osizes, no) = segs
    if nr == 1 && no == 1
        return g((rstrides[1],), (rsizes[1],), (ostrides[1],), (osizes[1],))
    elseif nr == 1 && no == 0
        return g((rstrides[1],), (rsizes[1],), (), ())
    else
        return g(_segment_tuple(rstrides, nr), _segment_tuple(rsizes, nr),
                 _segment_tuple(ostrides, no), _segment_tuple(osizes, no))
    end
end
_segment_tuple(t::Tuple, n::Int) = ntuple(i -> t[i], n)

# The source of a reduction along `dims` into a single output as a linearly indexed collection
# of its elements, for the whole-array kernels (`mapreduce_1d_gpu`): a `Broadcasted` object, an
# array without a dense buffer, or a contiguous range of a dense buffer; `nothing` for other
# sources.
_mapreduce_nd_linear(src::Base.Broadcast.Broadcasted, dims_valid) = src
function _mapreduce_nd_linear(src::AbstractArray, dims_valid)
    layout = _mapreduce_strided_layout(src)
    isnothing(layout) && return src
    buffer, base_offset, src_strides = layout
    (rstrides, _, nr), _ = _mapreduce_segments(size(src), src_strides, dims_valid)
    nr == 1 && rstrides[1] == 1 || return nothing
    return base_offset == 0 && buffer === src ? src :
           view(buffer, base_offset + 1:base_offset + length(src))
end

# (the strides of a dense column-major array: 1 and the running products of the sizes)
_mapreduce_dense_strides(sizes::Tuple{}) = ()
_mapreduce_dense_strides(sizes::Tuple) = Base.front(_cumprod((1, sizes...)))
_cumprod(t::Tuple{Any}) = t
_cumprod(t::Tuple) = (t[1], _cumprod((t[1] * t[2], Base.tail(Base.tail(t))...))...)

# The layout the reduction kernels index `src` with, as `(buffer, base_offset, strides)`: the
# strided layout of a source backed by a dense buffer, else (for a lazy array or a `Broadcasted`
# object) the source indexed linearly, as a dense array of its size. Indexing a lazy source
# linearly costs a decode of the index per element, but it gets the launch shapes of a dense
# array, which can spread a reduction into few outputs across many threads.
function _mapreduce_layout(src)
    layout = _mapreduce_strided_layout(src)
    return isnothing(layout) ? _mapreduce_linear_layout(src) : layout
end
_mapreduce_linear_layout(src) =
    (_mapreduce_linear_source(src), 0, _mapreduce_dense_strides(size(src)))
# (a reshaped array decodes a linear index with multiplicative inverses rather than divisions)
_mapreduce_linear_source(src::AbstractArray) = vec(src)
_mapreduce_linear_source(src::Base.Broadcast.Broadcasted) = src

# Resolve the layout the stride-based fast-path kernels need. Those kernels index a
# flat buffer at `buffer[base_offset + Σ coordᵈ·strideᵈ + 1]`, so they work for any
# source backed by a single dense column-major buffer — a dense array, but also a
# strided view, adjoint, permuted-dims, or reshape over one. Returns
# `(buffer, base_offset, strides)` for such sources, or `nothing` (→ linear indexing, see
# `_mapreduce_layout`) for `Broadcasted` and anything not backed by a dense buffer
# (lazy/computed arrays, complex adjoints without `strides`, nested wrappers).
function _mapreduce_strided_layout(src::AbstractArray)
    s = try
        strides(src)
    catch err
        # (an `ArgumentError` for a reshape of a non-contiguous view, on some Julia versions)
        err isa Union{MethodError, ArgumentError} || rethrow()
        return nothing
    end

    # Dense or contiguous (column-major) — index the source directly, no offset.
    s == _mapreduce_dense_strides(size(src)) && return (src, 0, s)

    # Non-dense but strided: index the dense parent buffer at the source's offset.
    # Only one level of wrapping over a dense buffer is supported; deeper nesting
    # falls back to the generic kernel.
    p = parent(src)
    (p === src || !_mapreduce_is_dense_buffer(p)) && return nothing
    base = _mapreduce_wrapper_offset(src, p)
    base === nothing && return nothing
    return (p, base, s)
end

# (before Julia 1.12 a `Broadcasted` source reaches the kernels materialized, as a dense array:
# see `_mapreduce_source`)
_mapreduce_strided_layout(src::Base.Broadcast.Broadcasted) =
    VERSION < v"1.12-" ? (nothing, 0, _mapreduce_dense_strides(size(src))) : nothing

function _mapreduce_is_dense_buffer(p::AbstractArray)
    try
        return strides(p) == _mapreduce_dense_strides(size(p))
    catch err
        # (an `ArgumentError` for a reshape of a non-contiguous view, on some Julia versions)
        err isa Union{MethodError, ArgumentError} || rethrow()
        return false
    end
end

# Offset (0-based, in elements) of the wrapper's first element within its dense
# parent buffer. SubArrays start at their first selected index; permuted-dims,
# reshapes, and (real) adjoints/transposes share the parent's first element.
function _mapreduce_wrapper_offset(src::SubArray, p)
    try
        # Base.map (not AK's array `map`, which shadows it inside this module).
        first_index = Base.map(first, parentindices(src))
        return LinearIndices(p)[first_index...] - 1
    catch
        return nothing   # exotic index types → fall back to the generic kernel
    end
end
_mapreduce_wrapper_offset(src, p) = 0

function _buffer_is_16byte_aligned(buffer)
    applicable(pointer, buffer) || return false
    ptr = try
        pointer(buffer)
    catch
        return false
    end
    applicable(UInt, ptr) || return false
    return UInt(ptr) % 16 == 0
end

function _contiguous_vector_width(
    buffer, base_offset, outer_strides, reduce_strides, reduce_size, block_size,
)
    T = eltype(buffer)
    isprimitivetype(T) && sizeof(T) in (4, 8) || return 0

    W = 16 ÷ sizeof(T)
    reduce_strides == (1,) || return 0
    base_offset % W == 0 || return 0
    Base.all(s -> s % W == 0, outer_strides) || return 0
    _buffer_is_16byte_aligned(buffer) || return 0

    # The vector kernel uses UInt32 indices. Leave room for the final grid-stride
    # increment so it cannot wrap after the last vector load.
    UInt128(reduce_size) + UInt128(block_size * W) <= typemax(UInt32) || return 0
    return W
end

# The reduced-extent index space: full axes along reduced dimensions, a single
# index (`OneTo(1)`) along kept dimensions. Combined with `max(Iother, Ireduce)`
# this reproduces Base's `mapreducedim!` iteration without touching strides.
_mapreduce_reduce_indices(src, dst) =
    CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))
