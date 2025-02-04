@kernel inbounds=true cpu=false function _mapreduce_block!(@Const(src), dst, f, op, init)

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
        sdata[ithread + 0x1] = init
    elseif i + N >= len
        sdata[ithread + 0x1] = f(src[i + 0x1])
    else
        sdata[ithread + 0x1] = op(f(src[i + 0x1]), f(src[i + N + 0x1]))
    end

    @synchronize()

    if N >= 512u16
        if ithread < 256u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 256u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 256u16
        if ithread < 128u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 128u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 128u16
        if ithread < 64u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 64u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 64u16
        if ithread < 32u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 32u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 32u16
        if ithread < 16u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 16u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 16u16
        if ithread < 8u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 8u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 8u16
        if ithread < 4u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 4u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 4u16
        if ithread < 2u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 2u16 + 0x1])
        end
        @synchronize()
    end
    if N >= 2u16
        if ithread < 1u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 0x1 + 0x1])
        end
        @synchronize()
    end

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
        dst[iblock + 0x1] = sdata[0x1]
    end
end



function mapreduce_1d(
    f, op, src::AbstractArray, backend::GPU;
    init,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    @argcheck 1 <= block_size <= 1024
    @argcheck switch_below >= 0

    # Degenerate cases
    len = length(src)
    len == 0 && return init
    len == 1 && return @allowscalar f(src[1])
    if len < switch_below
        h_src = Vector(src)
        return Base.mapreduce(f, op, h_src, init=init)
    end

    # Figure out type for destination
    dst_type = typeof(init)

    # Each thread will handle two elements
    num_per_block = 2 * block_size
    blocks = (len + num_per_block - 1) ÷ num_per_block

    if !isnothing(temp)
        @argcheck get_backend(temp) === get_backend(src)
        @argcheck eltype(temp) === dst_type
        @argcheck length(temp) >= blocks * 2
        dst = temp
    else
        dst = similar(src, dst_type, blocks * 2)
    end

    # Later the kernel will be compiled for views anyways, so use same types
    src_view = @view src[1:end]
    dst_view = @view dst[1:blocks]

    kernel! = _mapreduce_block!(backend, block_size)
    kernel!(src_view, dst_view, f, op, init, ndrange=(block_size * blocks,))

    # As long as we still have blocks to process, swap between the src and dst pointers at
    # the beginning of the first and second halves of dst
    len = blocks
    if len < switch_below
        h_src = Vector(@view(dst[1:len]))
        return Base.mapreduce(f, op, h_src, init=init)
    end

    # Now all src elements have been passed through f; just do final reduction, no map needed
    kernel2! = _reduce_block!(backend, block_size)
    p1 = @view dst[1:len]
    p2 = @view dst[blocks + 1:end]

    while len > 1
        blocks = (len + num_per_block - 1) ÷ num_per_block

        # Each block produces one reduced value
        kernel2!(p1, p2, op, init, ndrange=(block_size * blocks,))
        len = blocks

        if len < switch_below
            h_src = Vector(@view(p2[1:len]))
            return Base.reduce(op, h_src, init=init)
        end

        p1, p2 = p2, p1
        p1 = @view p1[1:len]
    end

    return @allowscalar p1[1]
end
