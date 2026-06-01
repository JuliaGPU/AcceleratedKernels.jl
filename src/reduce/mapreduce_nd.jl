function mapreduce_nd(
    f, op, src::AbstractArray, backend::Backend;
    init,
    neutral=neutral_element(op, eltype(src)),
    dims::Union{Int, Tuple{Vararg{Int}}},

    # CPU settings - ignored here
    max_tasks::Int,
    min_elems::Int,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int,
    temp::Union{Nothing, AbstractArray},
)
    @argcheck 1 <= block_size <= 1024

    # Degenerate cases begin; order of priority matters

    # Invalid dims
    if Base.any(d < 1 for d in (dims isa Int ? (dims,) : dims))
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # If dims > number of dimensions, just map each element through f and add init, e.g.:
    #   julia> x = rand(Float64, 3, 5);
    #   julia> mapreduce(x -> -x, +, x, dims=3, init=Float32(0))
    #   3×5 Matrix{Float32}     # Negative numbers
    src_sizes = size(src)
    if Base.all(d > length(src_sizes) for d in (dims isa Int ? (dims,) : dims))
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), src_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i in dims ? 1 : src_sizes[i]
    end

    # If any dimension except dims is zero, return empty similar array except with the dims
    # dimension = 1. Weird, see example below:
    #   julia> x = rand(3, 0, 5);
    #   julia> reduce(+, x, dims=3)
    #   3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize in dims && continue
        if src_sizes[isize] == 0
            if isnothing(temp)
                dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
            else
                @argcheck size(temp) == dst_sizes
                @argcheck eltype(temp) == typeof(init)
                dst = temp
            end
            return dst
        end
    end

    # If sizes[dims] == 0, return array filled with init; same shape except sizes[dims] = 1:
    #   julia> x = rand(3, 0, 5);
    #   julia> mapreduce(+, x, dims=2)
    #   3×1×5 Array{Float64, 3}:
    #   [:, :, 1] =
    #    0.0
    #    0.0
    #    0.0
    #   [...]
    len = Base.prod(src_sizes[d] for d in (dims isa Int ? (dims,) : dims))
    if len == 0
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == dst_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        fill!(dst, init)
        return dst
    end

    # If sizes[dims] == 1, just map each element through f. Again, keep same type as init
    if len == 1
        if isnothing(temp)
            dst = KernelAbstractions.allocate(backend, typeof(init), src_sizes)
        else
            @argcheck get_backend(temp) == backend
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # Degenerate cases end

    # Allocate destination array
    if isnothing(temp)
        dst = KernelAbstractions.allocate(backend, typeof(init), dst_sizes)
    else
        @argcheck get_backend(temp) == backend
        @argcheck size(temp) == dst_sizes
        @argcheck eltype(temp) == typeof(init)
        dst = temp
    end
    dst_size = length(dst)

    if !use_gpu_algorithm(backend, prefer_threads)
        _mapreduce_nd_cpu_sections!(
            f, op, dst, src;
            init,
            max_tasks=max_tasks,
            min_elems=min_elems,
        )
    else
        Rother  = CartesianIndices(dst)
        Rreduce = CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))

        if dst_size >= len
            blocks = (dst_size + block_size - 1) ÷ block_size
            kernel1! = _mapreduce_nd_by_thread!(backend, block_size)
            kernel1!(
                src, dst, f, op, init, Rother, Rreduce,
                ndrange=(block_size * blocks,),
            )
        else
            blocks = dst_size
            kernel2! = _mapreduce_nd_by_block!(backend, block_size)
            kernel2!(
                src, dst, f, op, init, neutral, Rother, Rreduce,
                ndrange=(block_size * blocks,),
            )
        end
    end
    return dst
end

function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init,
    max_tasks, min_elems,
)
    Rother  = CartesianIndices(dst)
    Rreduce = CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))

    foreachindex(dst, max_tasks=max_tasks, min_elems=min_elems) do idst
        @inbounds begin
            Iother = Rother[idst]
            res = init
            for Ireduce in Rreduce
                J = max(Iother, Ireduce)
                res = op(res, f(src[J]))
            end
            dst[idst] = res
        end
    end

    dst
end

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread!(
    @Const(src), dst,
    f, op,
    init, Rother, Rreduce,
)
    # One thread per output element, when there are more outer elements than in the reduced dims
    output_size = length(Rother)
    reduce_size = length(Rreduce)

    N = @groupsize()[1]

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    tid = ithread + iblock * N
    if tid < output_size
        Iother = Rother[tid + 0x1]

        res = init
        for i in 0x1:reduce_size
            Ireduce = Rreduce[i]
            J = max(Iother, Ireduce)
            res = op(res, f(src[J]))
        end
        dst[Iother] = res
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_block!(
    @Const(src), dst,
    f, op,
    init, neutral,
    Rother, Rreduce,
)
    # One block per output element, when there are more elements in the reduced dims than outer
    # e.g. reduce(+, rand(3, 1000), dims=2) => only 3 elements in outer dimensions
    reduce_size = length(Rreduce)

    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block handles one output element
    Iother = Rother[iblock + 0x1]

    # Pre-reduce in strides of N across the reduction space
    partial = neutral
    i = ithread + 0x1
    while i <= reduce_size
        Ireduce = Rreduce[i]
        J = max(Iother, Ireduce)
        partial = op(partial, f(src[J]))
        i += N
    end

    # Store partial result in shared memory and reduce within block
    sdata[ithread + 0x1] = partial
    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    if ithread == 0x0
        dst[Iother] = op(init, sdata[0x1])
    end
end
