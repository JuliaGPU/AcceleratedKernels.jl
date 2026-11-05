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
`threads_min_elems` for `CPUThreads.Partitioned`.
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
tuning_type(::typeof(findall_tuning)) = FindallTuning

"""
    resolve_findall(alg, backend, T) -> Algorithm

Resolve `alg` for `findall` over elements of type `T` on `backend`: `ScanScatter` or
`CPUThreads.Partitioned` with every field set, or an `ArgumentError`.
"""
function resolve_findall(alg::Algorithm, backend::Backend, ::Type{T}) where {T}
    t = tuning(findall_tuning, backend, T)
    a = alg isa Auto ? (runs_threads(backend) ? CPUThreads.Partitioned() : ScanScatter()) : alg
    a = fill_defaults(a, t, T)
    check_findall(a, backend)
    return a
end

function check_fields(a::ScanScatter)
    check_block_size(a)
    check_positive(a, :items_per_thread)
    nothing
end

fill_defaults(a::ScanScatter, t::FindallTuning, T) =
    ScanScatter(something(a.block_size, t.block_size),
                something(a.items_per_thread, t.items_per_thread))
fill_defaults(a::CPUThreads.Partitioned, t::FindallTuning, T) = fill_threads(a, t)
fill_defaults(a::Algorithm, t::FindallTuning, T) =
    throw(ArgumentError("$(algname(a)) is not a findall algorithm"))

function check_findall(a::ScanScatter, backend)
    check_fields(a)
    require_kernels(a, backend)
    check_tile(a)
    nothing
end

function check_findall(a::CPUThreads.Partitioned, backend)
    check_fields(a)
    require_threads(a, backend)
end


# The element of `indices` at the ordinal `position` (1-based, whatever the axes)
@inline ordinal_index(indices::Union{AbstractUnitRange, LinearIndices{1}}, position) =
    first(indices) + position - 1
@inline ordinal_index(indices, position) = @inbounds indices[firstindex(indices) + position - 1]


# Scatter the selected elements of each tile, from `block_counts`, the inclusive scan of the
# tiles' counts. A tile's count fits an `Int32`, so the block scans those.
@kernel cpu=false inbounds=true unsafe_indices=true function compact_kernel!(
    out, @Const(bools), block_counts, input_indices, output_indices, ::Val{ITEMS},
    subgroups::Val,
) where ITEMS
    @uniform block_size = @groupsize()[1]
    tile = @localmem UInt8 (tile_length(UInt8, block_size, ITEMS),)
    thread_counts = @localmem Int32 (block_size,)

    len = length(bools)
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    block_offset = iblock * block_size * ITEMS

    tile_load!(tile, bools, b -> UInt8(b), 0x0, k -> ordinal_index(input_indices, k + 0x1),
               block_offset, len, ithread, block_size, Val(ITEMS))
    @synchronize()

    run = ithread * ITEMS
    slot = ithread * tile_stride(UInt8, ITEMS)
    count = Int32(0)
    k = 0
    while k < ITEMS
        count += tile[slot + k + 0x1]
        k += 1
    end
    thread_counts[ithread + 0x1] = count

    thread_offset, _ = block_exclusive_scan!(
        @context, +, thread_counts, Int32(0), block_size, ithread, subgroups,
    )
    pos = (iblock == 0x0 ? 0 : block_counts[iblock]) + thread_offset

    k = 0
    while k < ITEMS
        if tile[slot + k + 0x1] != 0x0
            pos += 1
            position = block_offset + run + k + 0x1
            out[pos] = ordinal_index(output_indices, position)
        end
        k += 1
    end
end


# Stream compaction: the elements of `output_indices`, of type `I`, at the positions where
# `bools` is `true`. On GPUs, the result is an array like `like` (the searched array, rather than
# a mask of it in scratch memory), so that it shares properties such as the storage mode.
function compact(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend, alg::ScanScatter;
    bufs, like,
) where I
    isempty(bools) && return similar_on(backend, like, I, (0,))
    block_size = alg.block_size
    compact_tiles(bools, I, output_indices, backend, Val(block_size), Val(alg.items_per_thread),
                  scan_subgroup_width(backend, Int32, block_size), bufs, like)
end

# (a function barrier on the block size: from KernelAbstractions 0.10, constructing the kernels
# then infers, which saved 0.5 µs per kernel and call on CUDA)
function compact_tiles(bools, ::Type{I}, output_indices, backend, ::Val{B}, items::Val{N},
                       subgroups, bufs, like) where {I, B, N}
    num_blocks = cld(length(bools), B * N)
    block_counts = bufs.counts

    # Count each tile's selected elements, as the scans reduce their tiles, then scan the counts
    tile_aggregates_kernel!(backend, B)(
        +, block_counts, one_based(bools), b -> Int32(b), Int32(0), Int32(0), items, subgroups;
        ndrange=num_blocks * B)
    accumulate_nested!(+, block_counts, bufs.scan; backend, init=0)
    n = @allowscalar block_counts[end]

    out = similar_on(backend, like, I, (n,))
    if n > 0
        compact_kernel!(backend, B)(
            out, bools, block_counts, eachindex(bools), output_indices, items, subgroups;
            ndrange=num_blocks * B)
    end
    out
end


function compact(
    bools::AbstractArray{Bool}, ::Type{I}, output_indices, backend::Backend,
    alg::CPUThreads.Partitioned;
    bufs, like,
) where I
    input_indices = eachindex(bools)
    tp = TaskPartitioner(length(bools), alg.max_tasks, alg.min_elems)
    task_counts = bufs.counts
    itask_partition(tp) do itask, positions
        task_counts[itask] = Base.count(
            position -> @inbounds(bools[ordinal_index(input_indices, position)]), positions,
        )
    end
    cumsum!(task_counts, task_counts)

    out = similar(bools, I, task_counts[end])
    itask_partition(tp) do itask, positions
        offset = itask == 1 ? 0 : task_counts[itask - 1]
        compact_serial!(out, bools, input_indices, output_indices, positions, offset)
    end
    out
end


function compact_serial!(out, bools, input_indices, output_indices, positions, pos)
    @inbounds for position in positions
        if bools[ordinal_index(input_indices, position)]
            pos += 1
            out[pos] = ordinal_index(output_indices, position)
        end
    end
    out
end


function findall_mask!(bools, pred, v::AbstractArray, backend::Backend, alg)
    input_indices = eachindex(v)
    bool_indices = eachindex(bools)
    foreachindex_on(Base.OneTo(length(v)), backend; launch_kwargs(alg)...) do position
        input_index = ordinal_index(input_indices, position)
        bool_index = ordinal_index(bool_indices, position)
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
    values isa AbstractArray{Bool} || check_bool_result(identity, values)
    s = findall_setup(values, items, false, backend, alg)
    findall_run(identity, values, items, s, scratch_buffers(s.plan, workspace, values, items))
end

function findall(pred::F, v::AbstractArray; items::AbstractArray=keys(v),
                 backend::Union{Nothing, Backend}=nothing, alg::Algorithm=Auto(),
                 workspace=nothing) where {F}
    check_bool_result(pred, v)
    s = findall_setup(v, items, true, backend, alg)
    findall_run(pred, v, items, s, scratch_buffers(s.plan, workspace, v, items))
end

plan(::typeof(findall), values::AbstractArray; items::AbstractArray=keys(values),
     backend=nothing, alg::Algorithm=Auto()) =
    findall_setup(values, items, false, backend, alg).plan
plan(::typeof(findall), pred::F, v::AbstractArray; items::AbstractArray=keys(v),
     backend=nothing, alg::Algorithm=Auto()) where {F} =
    findall_setup(v, items, true, backend, alg).plan

# The plan of `findall` over `v`: its scratch is a mask of `pred(v[i])` for the predicate form
# and for non-Bool values, the counts of the blocks or tasks, and the scan of the block counts
function findall_setup(v, items, predicate::Bool, backend, alg)
    length(items) == length(v) || throw(DimensionMismatch(
        "`items` must have the length of the array, $(length(v)), got $(length(items))"))
    backend = resolve_backend(backend, v, items)
    a = resolve_findall(alg, backend, eltype(v))
    n = length(v)
    mask = predicate || !(v isa AbstractArray{Bool}) ? (; mask=scratch(Bool, size(v))) : (;)
    counts = if a isa ScanScatter
        num_blocks = cld(n, a.block_size * a.items_per_thread)
        (; counts=scratch(Int, num_blocks),
         scan=scan_setup(+, Int, Int, (num_blocks,), backend; init=0).plan)
    else
        (; counts=scratch(Int, task_count(n, a.max_tasks, a.min_elems)))
    end
    return (; plan=Plan(backend, a, merge(mask, counts)))
end

function findall_run(pred::F, v, items, s, bufs) where {F}
    backend, a = s.plan.backend, s.plan.alg
    bools = if haskey(bufs, :mask)
        findall_mask!(bufs.mask, pred, v, backend, a)
    else
        v
    end
    compact(bools, eltype(items), items, backend, a; bufs, like=v)
end
