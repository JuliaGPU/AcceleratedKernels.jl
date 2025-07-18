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
