"""
    ScanScatter(; block_size=nothing, items_per_thread=nothing)

Stable GPU stream compaction: each block counts the selected elements of its tile of
`block_size * items_per_thread` elements (`block_size` a power of two up to 1024), the counts are
scanned, and a second pass scatters the selected indices.
"""
Base.@kwdef struct ScanScatter <: FindallAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end


"""
    FindallTuning(; kwargs...)

Values that fill unset algorithm fields for `findall` on one device, as returned by
[`findall_tuning`](@ref): `block_size` and `items_per_thread` for `ScanScatter`, and
`threads_min_elems` for `CPUThreads.Partitioned`. The defaults reproduce AK's historical
settings. Internal: the fields may change in any release.
"""
Base.@kwdef struct FindallTuning
    block_size::Int = 256
    items_per_thread::Int = 16
    threads_min_elems::Int = 1
end

"""
    findall_tuning(backend, T) -> FindallTuning

The `findall` tuning for element type `T` (the input's) on `backend`'s current device; see
[`sort_tuning`](@ref) for the conventions.
"""
findall_tuning(::Backend, ::Type) = FindallTuning()

"""
    _resolve_findall(alg, backend, T) -> Algorithm

Resolve `alg` for `findall` over elements of type `T` on `backend`: `ScanScatter` or
`CPUThreads.Partitioned` with every field set, or an `ArgumentError`.
"""
function _resolve_findall(alg::Algorithm, backend::Backend, ::Type{T}) where {T}
    _checkdomain(alg)
    t = findall_tuning(backend, T)
    a = alg isa Auto ? (_runs_threads(backend) ? CPUThreads.Partitioned() : ScanScatter()) : alg
    a = _fill(a, t, T)
    _check_findall(a, backend)
    return a
end

function _checkdomain(a::ScanScatter)
    _check_pow2(a, :block_size)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "ScanScatter: `block_size` must be at most 1024, got $(a.block_size)"))
    _check_positive(a, :items_per_thread)
    nothing
end

_fill(a::ScanScatter, t::FindallTuning, T) =
    ScanScatter(something(a.block_size, t.block_size),
                something(a.items_per_thread, t.items_per_thread))
_fill(a::CPUThreads.Partitioned, t::FindallTuning, T) = _fill_threads(a, t)
_fill(a::Algorithm, t::FindallTuning, T) =
    throw(ArgumentError("$(_algname(a)) is not a findall algorithm"))

function _check_findall(a::ScanScatter, backend)
    _checkdomain(a)
    _require_kernels(a, backend)
    widemul(a.block_size, a.items_per_thread) <= typemax(Int32) || throw(ArgumentError(
        "ScanScatter: `block_size * items_per_thread` must be at most $(typemax(Int32))"))
    nothing
end

function _check_findall(a::CPUThreads.Partitioned, backend)
    _checkdomain(a)
    _require_threads(a, backend)
end


# The element of `indices` at the ordinal `position` (1-based, whatever the axes)
@inline findall_index(indices::AbstractUnitRange, position) =
    first(indices) + position - 1
@inline findall_index(indices::LinearIndices{1}, position) =
    first(indices) + position - 1
@inline findall_index(indices, position) = @inbounds indices[firstindex(indices) + position - 1]


# With `out === nothing`, compute block counts. Otherwise, `block_counts` contains their
# inclusive prefix scan and the kernel scatters the selected items.
@kernel cpu=false inbounds=true unsafe_indices=true function findall_block!(
    out, bools_arg, block_counts, input_indices, output_indices, ::Val{ITEMS},
) where ITEMS
    bools = _const_source(bools_arg)
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


function findall_gpu(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend, alg::ScanScatter;
    bufs,
) where I
    block_size = alg.block_size
    items_per_thread = alg.items_per_thread

    # The output on the resolved backend, which a range does not determine
    isempty(bools) && return KernelAbstractions.allocate(backend, I, 0)

    elems_per_block = block_size * items_per_thread
    num_blocks = cld(length(bools), elems_per_block)
    block_counts = bufs.counts
    input_indices = eachindex(bools)
    items = Val(items_per_thread)

    kernel! = findall_block!(backend, block_size)
    kernel!(nothing, bools, block_counts, input_indices, output_indices, items;
            ndrange=num_blocks * block_size)
    _accumulate_nested!(+, block_counts, bufs.scan; backend, init=0)
    n = @allowscalar block_counts[end]

    out = KernelAbstractions.allocate(backend, I, n)
    if n > 0
        kernel!(out, bools, block_counts, input_indices, output_indices, items;
                ndrange=num_blocks * block_size)
    end
    out
end


function findall_cpu(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend,
    alg::CPUThreads.Partitioned;
    bufs,
) where I
    input_indices = eachindex(bools)
    tp = TaskPartitioner(length(bools), alg.max_tasks, alg.min_elems)
    if tp.num_tasks == 1
        out = similar(bools, I, Base.count(bools))
        findall_section!(out, bools, input_indices, output_indices, Base.OneTo(length(bools)), 0)
        return out
    end

    task_counts = bufs.counts
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
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend, alg; bufs,
) where I
    if alg isa ScanScatter
        findall_gpu(bools, I, output_indices, backend, alg; bufs)
    else
        findall_cpu(bools, I, output_indices, backend, alg; bufs)
    end
end


# The launch settings of the mask pass for a resolved findall algorithm
_findall_launch(a::ScanScatter) = (; block_size=a.block_size)
_findall_launch(a::CPUThreads.Partitioned) = (; max_tasks=a.max_tasks, min_elems=a.min_elems)

function findall_bools!(bools, pred, v::AbstractArray, backend::Backend, alg)
    input_indices = eachindex(v)
    bool_indices = eachindex(bools)
    _foreachindex(Base.OneTo(length(v)), backend; _findall_launch(alg)...) do position
        input_index = findall_index(input_indices, position)
        bool_index = findall_index(bool_indices, position)
        @inbounds bools[bool_index] = pred(v[input_index]) ? true : false
    end
    bools
end


"""
    findall(A::AbstractArray; items=keys(A), backend=nothing, alg=Auto(), workspace=nothing)
    findall(pred, A::AbstractArray; items=keys(A), backend=nothing, alg=Auto(), workspace=nothing)

Stream compaction: select, in order, the elements of `items` at the positions where `A` is `true`,
or where `pred` returns `true` for `A`'s elements. Values used as conditions, and `pred`'s results,
must be `Bool`. The result is a new vector of `eltype(items)` on the backend.

Positions are ordinal: the `k`-th element of `A`, in the order of `eachindex(A)`, selects the
`k`-th element of `items`, which may be any array with `A`'s length (so offset axes are no
problem). What is selected is the caller's choice:

- `keys(A)` (the default) gives `A`'s indices: `Int`s for a vector, `CartesianIndex`es for
  other arrays, including 0-dimensional ones.
- `LinearIndices(A)` gives linear indices of any array.
- An array of values selects those values: `AK.findall(mask; items=A)` is `A[mask]` for a mask of
  `A`'s shape, in one pass instead of `findall` and a gather.

The supported inputs are arrays. Dictionaries, other iterables, and scalar inputs accepted by
`Base.findall` are outside the scope of this package.

`alg` is [`Auto()`](@ref Auto) by default: [`CPUThreads.Partitioned`](@ref
AcceleratedKernels.CPUThreads.Partitioned) on the host and [`ScanScatter`](@ref) on GPUs, with
the device's settings. `backend` is derived from `A` and `items`. `workspace` takes the scratch
memory of a [`workspace`](@ref) made for the same call (the mask of the predicate form and the
counts), so that `findall` allocates only its result.

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(Int32[5, -2, 8, -1, 3])
AK.findall(x -> x > 0, v)               # [1, 3, 5]
AK.findall(x -> x > 0, v; items=v)      # Int32[5, 8, 3]

m = CUDA.CuArray(Bool[1 0; 0 1])
AK.findall(m)                           # [CartesianIndex(1, 1), CartesianIndex(2, 2)]
AK.findall(m; items=LinearIndices(m))   # [1, 4]
```
"""
function findall(values::AbstractArray; items::AbstractArray=keys(values),
                 backend::Union{Nothing, Backend}=nothing, alg::Algorithm=Auto(),
                 workspace=nothing)
    values isa AbstractArray{Bool} || _check_bool_result(identity, values)
    s = _findall_setup(values, items, false, backend, alg)
    _findall_run(identity, values, items, s, _buffers(s.plan, workspace, values, items))
end

function findall(pred, v::AbstractArray; items::AbstractArray=keys(v),
                 backend::Union{Nothing, Backend}=nothing, alg::Algorithm=Auto(),
                 workspace=nothing)
    _check_bool_result(pred, v)
    s = _findall_setup(v, items, true, backend, alg)
    _findall_run(pred, v, items, s, _buffers(s.plan, workspace, v, items))
end

_plan(::typeof(findall), values::AbstractArray; items::AbstractArray=keys(values),
      backend=nothing, alg::Algorithm=Auto()) =
    _findall_setup(values, items, false, backend, alg).plan
_plan(::typeof(findall), pred, v::AbstractArray; items::AbstractArray=keys(v), backend=nothing,
      alg::Algorithm=Auto()) =
    _findall_setup(v, items, true, backend, alg).plan

# The plan of `findall` over `v`: its scratch is a mask of `pred(v[i])` for the predicate form
# and for non-Bool values, the counts of the blocks or tasks, and the scan of the block counts
function _findall_setup(v, items, predicate::Bool, backend, alg)
    length(items) == length(v) || throw(DimensionMismatch(
        "`items` must have the length of the array, $(length(v)), got $(length(items))"))
    backend = _resolve_backend(backend, v, items)
    a = _resolve_findall(alg, backend, eltype(v))
    n = length(v)
    mask = predicate || !(v isa AbstractArray{Bool}) ? (; mask=_buffer(Bool, size(v))) : (;)
    counts, nested = if a isa ScanScatter
        num_blocks = cld(n, a.block_size * a.items_per_thread)
        scan = _accumulate_setup(+, Int, Int, (num_blocks,), backend; init=0).plan
        (; counts=_buffer(Int, num_blocks), scan=scan.sizes), (; scan=scan.alg)
    else
        num_tasks = TaskPartitioner(n, a.max_tasks, a.min_elems).num_tasks
        (; counts=_buffer(Int, num_tasks)), (;)
    end
    return (; plan=_Plan(backend, a, nested, merge(mask, counts)))
end

function _findall_run(pred, v, items, s, bufs)
    backend, a = s.plan.backend, s.plan.alg
    bools = if haskey(bufs, :mask)
        findall_bools!(bufs.mask, pred, v, backend, a)
    else
        v
    end
    findall_impl(bools, eltype(items), items, backend, a; bufs)
end
