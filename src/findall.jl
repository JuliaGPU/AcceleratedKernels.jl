abstract type FindallAlgorithm end

"""
    ScanScatter(; block_size=256, items_per_thread=16)

Stable GPU stream compaction using per-block counts, a prefix scan, and a scatter pass.
`block_size` must be a power of two between 1 and 1024; `items_per_thread` must be positive.
"""
Base.@kwdef struct ScanScatter <: FindallAlgorithm
    block_size::Int = 256
    items_per_thread::Int = 16
end


findall_algorithm(alg::ScanScatter) = alg
function findall_algorithm(alg::FindallAlgorithm)
    throw(ArgumentError("$(typeof(alg)) is not supported by findall"))
end


@inline findall_index(indices::AbstractUnitRange, position) =
    first(indices) + position - 1
@inline findall_index(indices::LinearIndices{1}, position) =
    first(indices) + position - 1
@inline findall_index(indices, position) = @inbounds indices[position]


# With `out === nothing`, compute block counts. Otherwise, `block_counts` contains their
# inclusive prefix scan and the kernel scatters the selected indices.
@kernel cpu=false inbounds=true unsafe_indices=true function findall_block!(
    out, @Const(bools), block_counts, input_indices, output_indices, ::Val{ITEMS},
) where ITEMS
    @uniform block_size = @groupsize()[1]
    tile = @localmem UInt8 (block_size * ITEMS,)
    thread_counts = @localmem Int (block_size,)

    len = length(bools)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    j = 0
    while j < ITEMS
        p = j * block_size + ithread
        position = block_offset + p
        tile[p + 0x1] = position < len ?
                         UInt8(bools[findall_index(input_indices, position + 0x1)]) : 0x0
        j += 1
    end
    @synchronize()

    run = ithread * ITEMS
    count = 0
    k = 0
    while k < ITEMS
        count += tile[run + k + 0x1]
        k += 1
    end
    thread_counts[ithread + 0x1] = count

    seed = (isnothing(out) || iblock == 0x0) ? 0 : block_counts[iblock]
    pos, block_total = block_exclusive_scan!(
        @context, +, thread_counts, seed, block_size, ithread,
    )

    if isnothing(out)
        if ithread == 0x0
            block_counts[iblock + 0x1] = block_total
        end
    else
        k = 0
        while k < ITEMS
            if tile[run + k + 0x1] != 0x0
                pos += 1
                position = block_offset + run + k + 0x1
                out[pos] = findall_index(output_indices, position)
            end
            k += 1
        end
    end
end


function findall_temp(bools, backend, len, temp)
    if isnothing(temp)
        return KernelAbstractions.allocate(backend, Int, len)
    end

    @argcheck get_backend(temp) === backend
    @argcheck eltype(temp) === Int
    @argcheck length(temp) >= len
    @argcheck !Base.mightalias(temp, bools)
    view(temp, 1:len)
end


function findall_gpu(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend, alg::ScanScatter;
    temp,
) where I
    block_size = alg.block_size
    items_per_thread = alg.items_per_thread
    @argcheck 1 <= block_size <= 1024
    @argcheck ispow2(block_size)
    @argcheck items_per_thread > 0

    isempty(bools) && return similar(bools, I, 0)

    elems_per_block = block_size * items_per_thread
    num_blocks = cld(length(bools), elems_per_block)
    block_counts = findall_temp(bools, backend, num_blocks, temp)
    input_indices = eachindex(bools)
    items = Val(items_per_thread)

    kernel! = findall_block!(backend, block_size)
    kernel!(nothing, bools, block_counts, input_indices, output_indices, items;
            ndrange=num_blocks * block_size)
    accumulate!(+, block_counts, backend; init=0)
    n = @allowscalar block_counts[end]

    out = similar(bools, I, n)
    if n > 0
        kernel!(out, bools, block_counts, input_indices, output_indices, items;
                ndrange=num_blocks * block_size)
    end
    out
end


function findall_cpu(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend;
    max_tasks::Int,
    min_elems::Int,
    temp,
) where I
    input_indices = eachindex(bools)
    tp = TaskPartitioner(length(bools), max_tasks, min_elems)
    if tp.num_tasks == 1
        out = similar(bools, I, Base.count(bools))
        findall_section!(out, bools, input_indices, output_indices, Base.OneTo(length(bools)), 0)
        return out
    end

    task_counts = findall_temp(bools, backend, tp.num_tasks, temp)
    itask_partition(tp) do itask, positions
        task_counts[itask] = Base.count(
            position -> @inbounds(bools[findall_index(input_indices, position)]), positions,
        )
    end
    cumsum!(task_counts, task_counts)

    out = similar(bools, I, task_counts[end])
    itask_partition(tp) do itask, positions
        offset = itask == 1 ? 0 : task_counts[itask - 1]
        findall_section!(out, bools, input_indices, output_indices, positions, offset)
    end
    out
end


function findall_section!(out, bools, input_indices, output_indices, positions, pos)
    @inbounds for position in positions
        if bools[findall_index(input_indices, position)]
            pos += 1
            out[pos] = findall_index(output_indices, position)
        end
    end
    out
end


function findall_impl(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend;
    alg::FindallAlgorithm=ScanScatter(),
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    temp::Union{Nothing, AbstractArray}=nothing,
) where I
    alg = findall_algorithm(alg)

    if use_gpu_algorithm(backend, prefer_threads)
        findall_gpu(bools, I, output_indices, backend, alg; temp)
    else
        findall_cpu(bools, I, output_indices, backend; max_tasks, min_elems, temp)
    end
end


findall_output_indices(v, ::Type{Int}) = LinearIndices(v)
findall_output_indices(v, ::Type) = CartesianIndices(axes(v))


function findall_bools(
    pred, v::AbstractArray, backend::Backend, temp_bools;
    max_tasks, min_elems, prefer_threads, block_size,
)
    bools = if isnothing(temp_bools)
        similar(v, Bool)
    else
        @argcheck get_backend(temp_bools) === backend
        @argcheck eltype(temp_bools) === Bool
        @argcheck axes(temp_bools) == axes(v)
        @argcheck !Base.mightalias(temp_bools, v)
        temp_bools
    end
    input_indices = eachindex(v)
    bool_indices = eachindex(bools)
    foreachindex(Base.OneTo(length(v)), backend;
                 max_tasks, min_elems, prefer_threads, block_size) do position
        input_index = findall_index(input_indices, position)
        bool_index = findall_index(bool_indices, position)
        @inbounds bools[bool_index] = pred(v[input_index]) ? true : false
    end
    bools
end


"""
    findall(A::AbstractArray, backend::Backend=get_backend(A);
            alg::FindallAlgorithm=ScanScatter(),
            max_tasks::Int=Threads.nthreads(), min_elems::Int=1,
            prefer_threads::Bool=true,
            temp::Union{Nothing, AbstractArray}=nothing,
            temp_bools::Union{Nothing, AbstractArray}=nothing)
    findall(pred, A::AbstractArray, backend::Backend=get_backend(A);
            alg::FindallAlgorithm=ScanScatter(),
            max_tasks::Int=Threads.nthreads(), min_elems::Int=1,
            prefer_threads::Bool=true,
            temp::Union{Nothing, AbstractArray}=nothing,
            temp_bools::Union{Nothing, AbstractArray}=nothing)

Return the indices of the `true` elements of `A`, or of the elements for which `pred` returns
`true`, in the same order and with the same index types as `Base.findall`. Values used as
conditions must be `Bool`.

The supported inputs are arrays. Dictionaries, other iterables, and scalar inputs accepted by
`Base.findall` are outside the scope of this package.

## Settings

- `alg=ScanScatter()` selects the GPU algorithm and its tuning parameters.
- `max_tasks=Threads.nthreads()` and `min_elems=1` control CPU task partitioning.
- `temp=nothing` may provide the `Int` buffer used for block or task counts.
- `temp_bools=nothing` may provide the Bool mask for the predicate form or for a mask whose
  element type is not `Bool`. It must have the same axes as `A` and must not alias it.

On a GPU, `temp` needs at least
`cld(length(A), alg.block_size * alg.items_per_thread)` elements. On a CPU, it needs one element
per task used. Omitted buffers are allocated automatically.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(Int32[5, -2, 8, -1, 3])
AK.findall(x -> x > 0, v)               # [1, 3, 5]

m = CUDA.CuArray(Bool[1 0; 0 1])
AK.findall(m)                           # [CartesianIndex(1, 1), CartesianIndex(2, 2)]
```
"""
function findall(
    values::AbstractArray, backend::Backend=get_backend(values);
    alg::FindallAlgorithm=ScanScatter(),
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_bools::Union{Nothing, AbstractArray}=nothing,
)
    alg = findall_algorithm(alg)
    bools = if values isa AbstractArray{Bool}
        isnothing(temp_bools) ||
            throw(ArgumentError("temp_bools is not used for a Bool mask"))
        values
    else
        findall_bools(identity, values, backend, temp_bools;
                      max_tasks, min_elems, prefer_threads, block_size=alg.block_size)
    end
    I = keytype(values)
    output_indices = findall_output_indices(values, I)
    findall_impl(bools, I, output_indices, backend;
                 alg, max_tasks, min_elems, prefer_threads, temp)
end


function findall(
    pred, v::AbstractArray, backend::Backend=get_backend(v);
    alg::FindallAlgorithm=ScanScatter(),
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_bools::Union{Nothing, AbstractArray}=nothing,
)
    alg = findall_algorithm(alg)
    bools = findall_bools(pred, v, backend, temp_bools;
                          max_tasks, min_elems, prefer_threads, block_size=alg.block_size)
    I = ndims(v) == 0 ? Int : keytype(v)
    output_indices = findall_output_indices(v, I)
    findall_impl(bools, I, output_indices, backend;
                 alg, max_tasks, min_elems, prefer_threads, temp)
end
