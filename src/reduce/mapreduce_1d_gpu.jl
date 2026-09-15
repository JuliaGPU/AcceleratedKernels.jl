# _TypedMap: wraps f to ensure its output is type-stable as T.
# Uses @generated so the dispatch is resolved entirely at specialization time —
# no runtime dispatch, no boxing. Only instantiated when eltype(src) != dst_type.
struct _TypedMap{T, F}
    f::F
end

@generated function (m::_TypedMap{T, F})(x) where {T, F}
    # Determine at code-generation time which conversion to use
    RT = Core.Compiler.return_type(F, Tuple{eltype(x)})
    if RT === T
        # f already returns T: identity
        return :(m.f(x))
    elseif T <: Integer && RT <: Integer
        # Integer narrowing: use unchecked truncation
        return :(Base.unsafe_trunc(T, m.f(x)))
    elseif T <: AbstractFloat
        # Float cast: direct constructor (fptrunc/fpext, no checks)
        return :(T(m.f(x)))
    else
        # Composite types (tuples, structs): trust f returns T
        return :(m.f(x))
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_block!(@Const(src), dst, f, op, neutral)

    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    len = length(src)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    i = ithread + iblock * (N * 0x2)
    if i >= len
        @inbounds sdata[ithread + 0x1] = neutral
    elseif i + N >= len
        @inbounds sdata[ithread + 0x1] = f(@inbounds src[i + 0x1])
    else
        @inbounds sdata[ithread + 0x1] = op(f(@inbounds src[i + 0x1]), f(@inbounds src[i + N + 0x1]))
    end

    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    # Code below would work on NVidia GPUs with warp size of 32, but create race conditions and
    # return incorrect results on Intel Graphics. It would be useful to have a way to statically
    # query the warp size at compile time
    #
    # if ithread < 32
    #     N >= 64 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 32 + 1]))
    #     N >= 32 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 16 + 1]))
    #     N >= 16 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 8 + 1]))
    #     N >= 8 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 4 + 1]))
    #     N >= 4 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 2 + 1]))
    #     N >= 2 && (sdata[ithread + 1] = op(sdata[ithread + 1], sdata[ithread + 1 + 1]))
    # end

    if ithread == 0x0
        @inbounds dst[iblock + 0x1] = @inbounds sdata[0x1]
    end
end


function mapreduce_1d_gpu(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral,

    # CPU settings - ignored here
    max_tasks::Int,
    min_elems::Int,

    # GPU settings
    block_size::Int,
    temp::Union{Nothing, AbstractArray},
    switch_below::Int,
)
    @argcheck 1 <= block_size <= 1024
    @argcheck switch_below >= 0

    # Degenerate cases
    len = length(src)
    len == 0 && return init
    len == 1 && return @allowscalar f(src[1])
    if len < switch_below
        h_src = Vector(src)
        return Base.mapreduce(f, op, h_src; init)
    end

    # Each thread will handle two elements
    num_per_block = 2 * block_size
    blocks = (len + num_per_block - 1) ÷ num_per_block

    dst_type = typeof(init)
    if !isnothing(temp)
        @argcheck get_backend(temp) === backend
        @argcheck eltype(temp) === dst_type
        @argcheck length(temp) >= blocks * 2
        dst = temp
    else
        dst = KernelAbstractions.allocate(backend, dst_type, blocks * 2)
    end

    # Later the kernel will be compiled for views anyways, so use same types
    src_view = @view src[1:end]
    dst_view = @view dst[1:blocks]

    kernel! = _mapreduce_block!(backend, block_size)
    neutral_typed = convert(dst_type, neutral)
    # Only wrap f when output type conversion is needed; avoids overhead for same-type and composite types
    f_typed = (eltype(src_view) === dst_type) ? f : _TypedMap{dst_type, typeof(f)}(f)
    kernel!(src_view, dst_view, f_typed, op, neutral_typed, ndrange=(block_size * blocks,))

    # As long as we still have blocks to process, swap between the src and dst pointers at
    # the beginning of the first and second halves of dst
    len = blocks
    if len < switch_below
        h_src = Vector(@view(dst[1:len]))
        return Base.reduce(op, h_src; init)
    end

    # Now all src elements have been passed through f; just do final reduction, no map needed
    p1 = @view dst[1:len]
    p2 = @view dst[blocks + 1:end]

    while len > 1
        blocks = (len + num_per_block - 1) ÷ num_per_block

        # Each block produces one reduced value
        kernel!(p1, p2, identity, op, neutral_typed, ndrange=(block_size * blocks,))
        len = blocks

        if len < switch_below
            h_src = Vector(@view(p2[1:len]))
            return Base.reduce(op, h_src; init)
        end

        p1, p2 = p2, p1
        p1 = @view p1[1:len]
    end

    # The GPU kernel reduced all elements to one, but without the init value
    return op(init, @allowscalar(p1[1]))
end
