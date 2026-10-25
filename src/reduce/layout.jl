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

# Where the kernels find the elements of the source `src` of a reduction along `dims`, which
# `select_launch` records so that the run takes them from the source it reduces: `:parent`, the
# dense buffer `src` wraps (see `strided_layout`); `:linear`, `src` indexed linearly (see
# `linear_layout`), a lazy array or a `Broadcasted` object; or `:source`, the source itself, a
# dense array.
buffer_kind(src, strided) =
    isnothing(strided) ? :linear : strided.buffer === src ? :source : :parent

# The buffer the kernels index, from the source `src` and the `buffer_kind`
kernel_buffer(src, kind::Symbol) =
    kind === :parent ? parent(src) : kind === :linear ? linear_view(src) : src
# ... and whether it is a dense buffer, which the kernels can read with vector loads
isdense_buffer(src, kind::Symbol) = kind === :parent || kind === :source

# The source of a reduction along `dims` into a single output of a contiguous source, as a
# linearly indexed collection of its elements for the whole-array kernels (`mapreduce_1d_gpu`):
# the source itself, or the range of its elements in the dense buffer it wraps, from `offset` on
linear_source(src, kind::Symbol, offset::Int) =
    kind === :parent ? view(parent(src), offset + 1:offset + length(src)) : src

# (the strides of a dense column-major array: 1 and the running products of the sizes)
dense_strides(sizes::Tuple{}) = ()
dense_strides(sizes::Tuple) = Base.front(tuple_cumprod((1, sizes...)))
tuple_cumprod(t::Tuple{Any}) = t
tuple_cumprod(t::Tuple) = (t[1], tuple_cumprod((t[1] * t[2], Base.tail(Base.tail(t))...))...)

# The layout, as `(; offset, strides)`, of a source without a dense buffer (a lazy array or a
# `Broadcasted` object), which the kernels index linearly (`linear_view`), as a dense array of
# its size. Indexing a lazy source linearly costs a decode of the index per element, but it gets
# the launch shapes of a dense array, which can spread a reduction into few outputs across many
# threads.
linear_layout(src) = (; offset=0, strides=dense_strides(size(src)))
# (a reshaped array decodes a linear index with multiplicative inverses rather than divisions)
linear_view(src::AbstractArray) = vec(src)
linear_view(src::Base.Broadcast.Broadcasted) =
    length(src) <= typemax(UInt32) ? LinearBroadcast(src) : src

# A `Broadcasted` object indexed linearly, as a vector of its elements in column-major order.
# Each array argument is indexed with its own linear index, which takes no division for an
# argument of the broadcast's shape, and for another (a row of means, say) a division or a
# remainder by a multiplicative inverse per run of its kept dimensions. (The `Broadcasted`
# object itself decodes the broadcast's index into Cartesian indices, with a division per
# dimension, and computes each argument's index from those.) The index is 32-bit: a 64-bit
# multiplicative inverse takes 128-bit products, which SPIR-V lacks.
struct LinearBroadcast{T, F, Args <: Tuple} <: AbstractVector{T}
    f::F
    args::Args
    len::Int
    LinearBroadcast{T}(f::F, args::Args, len::Int) where {T, F, Args <: Tuple} =
        new{T, F, Args}(f, args, len)
end
LinearBroadcast(bc::Base.Broadcast.Broadcasted) = linear_broadcast(source_eltype(bc), bc, size(bc))
linear_broadcast(::Type{T}, bc, sz) where {T} =
    LinearBroadcast{T}(bc.f, linear_args(bc.args, sz), Base.prod(sz))
linear_args(args::Tuple, sz) = (linear_arg(args[1], sz), linear_args(Base.tail(args), sz)...)
linear_args(::Tuple{}, sz) = ()
Base.size(a::LinearBroadcast) = (a.len,)
Base.IndexStyle(::Type{<:LinearBroadcast}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::LinearBroadcast, i::Int) = evaluate(a, (i - 1) % UInt32)
@inline evaluate(a::LinearBroadcast, i::UInt32) = a.f(Base.map(x -> evaluate(x, i), a.args)...)
Adapt.adapt_structure(to, a::LinearBroadcast{T}) where {T} =
    LinearBroadcast{T}(Adapt.adapt(to, a.f), Adapt.adapt(to, a.args), a.len)

# An argument `x` of a broadcast, with the runs of consecutive dimensions it is not extruded
# along: from the broadcast's linear index `i`, a run contributes `rem(div(i, d), m) * stride`
# to `x`'s linear index, where `nothing` skips a division or remainder that would change
# nothing. An argument of the broadcast's shape has a single run without either. (The types of
# the runs depend on the sizes, so the kernels specialize on them: a division by a
# multiplicative inverse in every load, even one by 1, makes a reduction up to several times
# slower.) Scalars and `Ref`s have no runs.
struct LinearArg{X, R <: Tuple}
    x::X
    runs::R
end
Adapt.adapt_structure(to, a::LinearArg) = LinearArg(Adapt.adapt(to, a.x), a.runs)

linear_arg(bc::Base.Broadcast.Broadcasted, sz) = linear_broadcast(Any, bc, sz)
linear_arg(x::Base.Broadcast.Extruded, sz) = linear_arg(x.x, sz)
# (an argument may have offset axes where the broadcast's do not: one extruded along them)
function linear_arg(x::Union{AbstractArray, Tuple}, sz)
    runs, n = arg_runs(Base.map(length, axes(x)), sz)
    return LinearArg(one_based(x), specialize_runs(runs, n))
end
linear_arg(x, sz) = LinearArg(x, ())

const IndexInverse = Base.MultiplicativeInverses.UnsignedMultiplicativeInverse{UInt32}
specialize_runs(runs::Tuple, n) =
    n == 0 ? () : (specialize_run(runs[1]), specialize_runs(Base.tail(runs), n - 1)...)
specialize_run((d, m, stride)) =
    (d == 1 ? nothing : IndexInverse(d % UInt32), m == 0 ? nothing : IndexInverse(m % UInt32),
     stride)

@inline evaluate(a::LinearArg{<:Union{AbstractArray, Tuple}}, i::UInt32) =
    @inbounds a.x[arg_index(a.runs, i) + 1]
@inline evaluate(a::LinearArg, i::UInt32) = a.x[]

@inline arg_index(::Tuple{}, i) = 0
@inline function arg_index(runs::Tuple, i)
    d, m, stride = runs[1]
    return (run_rem(run_div(i, d), m) % Int) * stride + arg_index(Base.tail(runs), i)
end
@inline run_div(i, ::Nothing) = i
@inline run_div(i, d) = div(i, d)
@inline run_rem(i, ::Nothing) = i
@inline run_rem(i, m) = rem(i, m)

# The runs of `LinearArg` for an argument of size `asz` in a broadcast of size `sz`, each as
# `(d, m, stride)` with `m == 0` for no remainder, and their number. Dimensions where the
# broadcast has size 1 do not split a run.
function arg_runs(asz::Tuple, sz::NTuple{N, Int}) where {N}
    runs = ntuple(_ -> (1, 0, 0), Val(cld(N, 2)))
    kept(d) = sz[d] != 1 && d <= length(asz) && asz[d] != 1
    len = Base.prod(sz)
    n = 0
    before = 1    # the broadcast's elements per step along `d`
    abefore = 1   # the argument's
    d = 1
    while d <= N
        if !kept(d)
            before *= sz[d]
            d += 1
            continue
        end
        span, astride = 1, abefore
        while d <= N && (sz[d] == 1 || kept(d))
            span *= sz[d]
            abefore *= kept(d) ? asz[d] : 1
            d += 1
        end
        n += 1
        runs = Base.setindex(runs, (before, before * span == len ? 0 : span, astride), n)
        before *= span
    end
    return runs, n
end

# The layout of a source backed by a single dense column-major buffer, which the kernels index
# at `buffer[offset + Σ coordᵈ·strideᵈ + 1]`: a dense array, or a strided view, adjoint,
# permuted-dims array or reshape of one. `nothing` for other sources (lazy arrays, `Broadcasted`
# objects, complex adjoints, wrappers of wrappers), which are indexed linearly (see
# `linear_layout`).
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
