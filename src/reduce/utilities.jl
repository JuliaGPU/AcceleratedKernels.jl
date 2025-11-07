# neutral_element moved over from GPUArrays.jl
neutral_element(op, T) =
    error("""AcceleratedKernels.jl needs to know the neutral element for your operator `$op`.
             Please pass it as an explicit keyword argument `neutral`.""")
neutral_element(::typeof(Base.:(|)), T) = zero(T)
neutral_element(::typeof(Base.:(⊻)), T) = zero(T)
neutral_element(::typeof(Base.:(&)), T) = ~zero(T)
neutral_element(::typeof(Base.:(+)), T) = zero(T)
neutral_element(::typeof(Base.add_sum), T) = zero(T)
neutral_element(::typeof(Base.:(*)), T) = one(T)
neutral_element(::typeof(Base.mul_prod), T) = one(T)
neutral_element(::typeof(Base.min), T) = typemax(T)
neutral_element(::typeof(Base.max), T) = typemin(T)
neutral_element(::typeof(Base._extrema_rf), ::Type{<:NTuple{2,T}}) where {T} = typemax(T), typemin(T)


# Unrolled map constructing a tuple
@inline function unrolled_map_index(f, tuple_vector::Tuple)
    _unrolled_map_index(f, tuple_vector, (), 1)
end


@inline function _unrolled_map_index(f, rest::Tuple{}, acc, i)
    acc
end


@inline function _unrolled_map_index(f, rest::Tuple, acc, i)
    result = f(i)
    _unrolled_map_index(f, Base.tail(rest), (acc..., result), i + 1)
end


# Apply op(init, f(x)) to each element of src, storing the result in dst.
function _mapreduce_nd_apply_init!(
    f, op, dst, src, backend;
    init,
    max_tasks=Threads.nthreads(),
    min_elems=1,
    block_size=256,
)
    foreachindex(dst, backend; max_tasks, min_elems, block_size) do i
        dst[i] = op(init, f(src[i]))
    end
end

@inline function reduce_group!(op, sdata, N, ithread)
    if N >= 512u16
        if ithread < 256u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 256u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 256u16
        if ithread < 128u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 128u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 128u16
        if ithread < 64u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 64u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 64u16
        if ithread < 32u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 32u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 32u16
        if ithread < 16u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 16u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 16u16
        if ithread < 8u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 8u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 8u16
        if ithread < 4u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 4u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 4u16
        if ithread < 2u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 2u16 + 0x1])
        end
        KI.barrier()
    end
    if N >= 2u16
        if ithread < 1u16
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 1u16 + 0x1])
        end
        KI.barrier()
    end
end
