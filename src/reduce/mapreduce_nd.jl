# Generalized N-dimensional mapreduce for GPU and CPU backends, reducing one or more
# dimensions (`dims::Integer` or a collection of integers) of `src` into `dst`.
#
# Design (see references: CUDA.jl / GPUArrays mapreducedim!, PyTorch Reduce.cuh, CUB):
#   1. Canonicalize dims: collapse adjacent dimensions with matching strides into contiguous
#      segments. A plain `dims=2` reduction, or any reduction over a contiguous block of dims
#      (e.g. `dims=(1,2)`), collapses to a *single* reduce segment.
#   2. The inner reduce loop must avoid per-element integer division. For a single segment the
#      element offset is just `j * stride`; only genuinely non-contiguous dim sets (e.g.
#      `dims=(1,3)`) fall back to a per-element multi-dimensional decode.
#   3. Four work decompositions, chosen by the relative sizes and strides of the output and
#      the reduction:
#        - by_thread:     one thread per output (many outputs, small reduction)
#        - tiled_strided: several contiguous outputs per block, for square-like strided reductions
#        - by_block:      one block per output, grid-stride over outputs (few outputs, large reduction)
#        - multigroup:    several blocks per output, two-pass (dst_size==1 or very small dst_size)

# Number of blocks the by_block / multigroup paths aim to launch, so a reduction with
# few output elements can still fill the GPU. A heuristic GPU-occupancy target.
const TARGET_BLOCKS = 256

# Below this many output elements, splitting a single output's reduction across multiple
# blocks (multigroup) is preferred over grid-striding by_block, because grid-stride with
# too few blocks cannot fill the GPU on its own. At or above this, by_block grid-strides.
const GS_DST_CUTOFF = 32

# Minimum reduction-loop iterations per thread in the multigroup first pass.
# Caps reduce_groups so splitting a reduction doesn't shrink per-thread work
# below the point where launch/scheduling overhead dominates actual work.
const MIN_ITEMS_PER_THREAD = 8

# Number of contiguous output rows handled by each tiled-strided block. With the default
# 256-thread block this gives 32 lanes per row.
const TILED_STRIDED_ROWS_PER_BLOCK = 8


# Host-side canonicalization: split the dimensions into reduced and kept ("outer") segments,
# merging adjacent dimensions whose strides are contiguous. Returns two tuples of
# (stride, size) segments.
function _canonicalize_dims(src_sizes, src_strides, dims_valid)
    ndim = length(src_sizes)
    reduce_segs = Tuple{Int,Int}[]
    outer_segs  = Tuple{Int,Int}[]

    i = 1
    while i <= ndim
        if i in dims_valid
            seg_stride = src_strides[i]
            seg_size   = src_sizes[i]
            while i + 1 <= ndim &&
                  (i + 1) in dims_valid &&
                  src_strides[i + 1] == seg_stride * seg_size
                i += 1
                seg_size *= src_sizes[i]
            end
            push!(reduce_segs, (seg_stride, seg_size))
        else
            seg_stride = src_strides[i]
            seg_size   = src_sizes[i]
            while i + 1 <= ndim &&
                  !((i + 1) in dims_valid) &&
                  src_strides[i + 1] == seg_stride * seg_size
                i += 1
                seg_size *= src_sizes[i]
            end
            push!(outer_segs, (seg_stride, seg_size))
        end
        i += 1
    end

    return Tuple(reduce_segs), Tuple(outer_segs)
end

# Main entry point

function mapreduce_nd(
    f, op, src::MapReduceSource, backend::Backend;
    init,
    neutral=neutral_element(op, typeof(init)),
    dims,

    # CPU settings
    max_tasks::Int,
    min_elems::Int,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int,
    temp::Union{Nothing, AbstractArray},
)
    @argcheck 1 <= block_size <= 1024
    @argcheck ispow2(block_size)

    dims_src = dims isa Number ? (dims,) : dims
    dims_buf = Int[]
    for d in dims_src
        d isa Integer || throw(ArgumentError("reduced dimension(s) must be integers"))
        dim = Int(d)
        dim < 1 && throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
        push!(dims_buf, dim)
    end

    # Match Base: duplicate dims are ignored, e.g. dims=(2,2) behaves like dims=2.
    dims_all = Tuple(Base.unique(dims_buf))

    src_sizes = size(src)
    ndim      = length(src_sizes)

    dims_valid = Tuple(d for d in dims_all if d <= ndim)

    # Degenerate cases begin; order of priority matters

    # All reduced dims are beyond ndims: just map each element through f and add init, e.g.:
    #   julia> x = rand(Float64, 3, 5);
    #   julia> mapreduce(x -> -x, +, x, dims=3, init=Float32(0))     # 3×5 Matrix{Float32}
    if isempty(dims_valid)
        dst = _alloc_or_temp(backend, temp, init, src_sizes)
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i in dims_valid ? 1 : src_sizes[i]
    end

    # If any kept dimension is zero, return empty array (reduced dims become 1), e.g.:
    #   julia> x = rand(3, 0, 5);  reduce(+, x, dims=3)     # 3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize in dims_valid && continue
        if src_sizes[isize] == 0
            return _alloc_or_temp(backend, temp, init, dst_sizes)
        end
    end

    len = Base.prod(src_sizes[d] for d in dims_valid)

    # If a reduced dimension is zero, return array filled with init, e.g.:
    #   julia> x = rand(3, 0, 5);  mapreduce(+, x, dims=2)     # 3×1×5 of zeros
    if len == 0
        dst = _alloc_or_temp(backend, temp, init, dst_sizes)
        fill!(dst, init)
        return dst
    end

    # If the reduced extent is 1, just map each element through f (keep init's type)
    if len == 1
        dst = _alloc_or_temp(backend, temp, init, src_sizes)
        _mapreduce_nd_apply_init!(f, op, dst, src, backend; init, max_tasks, min_elems, block_size)
        return dst
    end

    # Degenerate cases end

    dst = _alloc_or_temp(backend, temp, init, dst_sizes)
    dst_size = length(dst)

    if backend == CPU_BACKEND
        _mapreduce_nd_cpu_sections!(f, op, dst, src; init, max_tasks, min_elems)
        return dst
    end

    # The stride-based fast paths below index a flat buffer at
    # `buffer[base_offset + Σ coordᵈ·strideᵈ + 1]`. This works for any source backed
    # by a single dense column-major buffer (dense arrays, but also strided views,
    # adjoints, permuted dims, and reshapes over one). Sources without such a buffer
    # — `Broadcasted`, lazy/computed arrays — take the generic Cartesian-indexed
    # fallback, which makes no layout assumption (and crucially does not wrap the
    # source in `@Const`, which is what makes e.g. PermutedDimsArray uncompilable).
    layout = _mapreduce_strided_layout(src)
    if isnothing(layout)
        blocks = cld(dst_size, block_size)
        kernel! = _mapreduce_nd_generic!(backend, block_size)
        kernel!(
            src, dst, f, op, init,
            CartesianIndices(dst), _mapreduce_reduce_indices(src, dst), dst_size,
            ndrange=(block_size * blocks,),
        )
        return dst
    end

    buffer, base_offset, src_strides = layout
    reduce_segs, outer_segs = _canonicalize_dims(src_sizes, src_strides, dims_valid)

    outer_strides  = Tuple(str for (str, _) in outer_segs)
    outer_sizes    = Tuple(s   for (_, s)   in outer_segs)
    reduce_strides = Tuple(str for (str, _) in reduce_segs)
    reduce_sizes   = Tuple(s   for (_, s)   in reduce_segs)
    reduce_size    = len

    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch decision (see header comment for the four paths):
    #
    #   - square-like, contiguous output with strided input -> tiled_strided
    #   - dst_size >= reduce_size                          -> by_thread
    #   - dst_size == 1, or dst_size < GS_DST_CUTOFF
    #     (and dst_size < reduce_size)                     -> multigroup (split one
    #                                                          reduction across blocks,
    #                                                          needs a 2nd-pass combine)
    #   - otherwise (GS_DST_CUTOFF <= dst_size < reduce_size) -> by_block, grid-striding
    #                                                            over outputs, single pass
    #
    # Rationale: grid-striding by_block launches `min(dst_size, TARGET_BLOCKS)` blocks;
    # for very small dst_size (e.g. 5 or 9) that under-fills an 84-SM GPU, so splitting
    # the (large) reduction itself across many blocks via multigroup is still better.
    # For dst_size==1 there is nothing to grid-stride over, so multigroup is the only
    # option regardless of GS_DST_CUTOFF.
    # ─────────────────────────────────────────────────────────────────────────

    # by_block override 1: when the reduced dimension is the fastest-varying one
    # (reduce_strides==(1,)), by_thread's per-thread strided access is badly
    # uncoalesced, while by_block lets consecutive threads read consecutive elements.
    use_by_block_for_coalescing =
        dst_size >= reduce_size && reduce_size >= block_size &&
        length(reduce_sizes) == 1 && reduce_sizes[1] != 0 &&
        reduce_strides == (1,)

    # by_block override 2: when by_thread would launch too few blocks for a square-like
    # output/reduction shape, split each output reduction across a full block. This
    # fallback is mostly for layouts that do not satisfy the narrower tiled-strided
    # pattern below; the common contiguous-output strided case uses tiled_strided.
    # Avoid applying this to wide-output shapes, where by_thread's cross-output
    # coalescing is better than a strided block reduction.
    use_by_block_for_low_occupancy =
        dst_size == reduce_size && reduce_size >= block_size &&
        cld(dst_size, block_size) < TARGET_BLOCKS &&
        length(reduce_sizes) == 1 && reduce_sizes[1] != 0

    use_by_block = use_by_block_for_coalescing || use_by_block_for_low_occupancy
    use_tiled_strided =
        dst_size == reduce_size && reduce_size >= block_size &&
        length(outer_sizes) == 1 && outer_strides == (1,) &&
        length(reduce_sizes) == 1 && reduce_strides[1] > 1 &&
        block_size % TILED_STRIDED_ROWS_PER_BLOCK == 0 &&
        ispow2(block_size ÷ TILED_STRIDED_ROWS_PER_BLOCK)

    if use_tiled_strided
        # Narrow layout-specific path: square-like row reductions with contiguous
        # outputs and strided input reads, e.g. size=(1024,1024), dims=2. Several
        # outputs share a block: small lane groups reduce one output each, preserving
        # more cross-output memory coalescing than by_block while launching many more
        # blocks than by_thread. This is the only extra kernel beyond the generic
        # by_thread/by_block/multigroup shapes.
        rows_per_block = TILED_STRIDED_ROWS_PER_BLOCK
        blocks = cld(dst_size, rows_per_block)
        kernel! = _mapreduce_nd_by_thread_tiled_strided!(backend, block_size)
        kernel!(
            buffer, dst, f, op, init, neutral,
            base_offset, reduce_strides[1], dst_size, reduce_size, Val(rows_per_block),
            ndrange=(block_size * blocks,),
        )
    elseif dst_size >= reduce_size && !use_by_block
        # Many outputs, small reduction: one thread per output reduces sequentially
        blocks = cld(dst_size, block_size)
        kernel! = _mapreduce_nd_by_thread!(backend, block_size)
        kernel!(
            buffer, dst, f, op, init,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size,
            ndrange=(block_size * blocks,),
        )
    elseif use_by_block
        # Grid-stride by_block: one full block cooperates on each output, then
        # grid-strides across remaining outputs if fewer blocks than outputs were
        # launched.
        launch_blocks = min(dst_size, TARGET_BLOCKS)
        kernel! = _mapreduce_nd_by_block!(backend, block_size)
        kernel!(
            buffer, dst, f, op, init, neutral,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, launch_blocks,
            ndrange=(block_size * launch_blocks,),
        )
    elseif dst_size == 1 || dst_size < GS_DST_CUTOFF
        # Very few outputs, large reduction: split each output's reduction across
        # `reduce_groups` blocks (multigroup), combine partials in a second pass.
        reduce_groups = min(
            cld(reduce_size, block_size),
            block_size,
            cld(TARGET_BLOCKS, dst_size),
            cld(reduce_size, block_size * MIN_ITEMS_PER_THREAD),
        )
        reduce_groups = max(reduce_groups, 1)

        if reduce_groups > 1
            partial = KernelAbstractions.allocate(backend, typeof(init), (dst_size, reduce_groups))

            kernel! = _mapreduce_nd_multigroup!(backend, block_size)
            kernel!(
                buffer, partial, f, op, neutral,
                base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
                Val(dst_size), reduce_size, reduce_groups,
                ndrange=(block_size * dst_size * reduce_groups,),
            )

            # Second pass: reduce partial (dst_size × reduce_groups) → dst, one block
            # per output. reduce_groups is small (<=block_size by construction), so use
            # a small block size for this pass to avoid an oversubscribed tree-reduce.
            pass2_block_size = _pass2_block_size(reduce_groups)
            kernel2! = _mapreduce_partial_to_dst!(backend, pass2_block_size)
            kernel2!(
                partial, dst, op, init, neutral,
                dst_size, reduce_groups,
                ndrange=(pass2_block_size * dst_size,),
            )
        else
            # reduce_groups collapsed to 1 (e.g. reduce_size <= block_size): just do a
            # single-block-per-output reduction directly, no partial array needed.
            kernel! = _mapreduce_nd_by_block!(backend, block_size)
            kernel!(
                buffer, dst, f, op, init, neutral,
                base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
                dst_size, reduce_size, dst_size,
                ndrange=(block_size * dst_size,),
            )
        end
    else
        # GS_DST_CUTOFF <= dst_size < reduce_size: grid-stride over outputs, one pass.
        # Cap launched blocks at TARGET_BLOCKS; each block handles
        # ceil(dst_size / launch_blocks) outputs sequentially.
        launch_blocks = min(dst_size, TARGET_BLOCKS)
        kernel! = _mapreduce_nd_by_block!(backend, block_size)
        kernel!(
            buffer, dst, f, op, init, neutral,
            base_offset, outer_strides, outer_sizes, reduce_strides, reduce_sizes,
            dst_size, reduce_size, launch_blocks,
            ndrange=(block_size * launch_blocks,),
        )
    end

    return dst
end

function _mapreduce_dense_strides(sizes::Tuple)
    stride = 1
    return ntuple(length(sizes)) do i
        s = stride
        stride *= sizes[i]
        s
    end
end

# Resolve the layout the stride-based fast-path kernels need. Those kernels index a
# flat buffer at `buffer[base_offset + Σ coordᵈ·strideᵈ + 1]`, so they work for any
# source backed by a single dense column-major buffer — a dense array, but also a
# strided view, adjoint, permuted-dims, or reshape over one. Returns
# `(buffer, base_offset, strides)` for such sources, or `nothing` (→ generic
# Cartesian-indexed fallback) for `Broadcasted` and anything not backed by a dense
# buffer (lazy/computed arrays, complex adjoints without `strides`, nested wrappers).
function _mapreduce_strided_layout(src::AbstractArray)
    s = try
        strides(src)
    catch err
        err isa MethodError || rethrow()
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

_mapreduce_strided_layout(::Base.Broadcast.Broadcasted) = nothing

function _mapreduce_is_dense_buffer(p::AbstractArray)
    try
        return strides(p) == _mapreduce_dense_strides(size(p))
    catch err
        err isa MethodError || rethrow()
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

# The reduced-extent index space: full axes along reduced dimensions, a single
# index (`OneTo(1)`) along kept dimensions. Combined with `max(Iother, Ireduce)`
# this reproduces Base's `mapreducedim!` iteration without touching strides.
_mapreduce_reduce_indices(src, dst) =
    CartesianIndices(ifelse.(axes(src) .== axes(dst), Ref(Base.OneTo(1)), axes(src)))


# Smallest power-of-two >= n, capped at 256 (the original fixed block size) and
# floored at 32 (one warp) — used for the multigroup second pass, which only ever
# combines `reduce_groups` values (reduce_groups <= block_size by construction).
@inline function _pass2_block_size(n::Int)
    n <= 32  && return 32
    n <= 64  && return 64
    n <= 128 && return 128
    return 256
end


# Allocate a destination array, or validate and reuse a user-provided `temp`.
function _alloc_or_temp(backend, temp, init, sizes)
    isnothing(temp) && return KernelAbstractions.allocate(backend, typeof(init), sizes)
    @argcheck get_backend(temp) == backend
    @argcheck size(temp) == sizes
    @argcheck eltype(temp) == typeof(init)
    temp
end

# CPU path

function _mapreduce_nd_cpu_sections!(
    f, op, dst, src;
    init, max_tasks, min_elems,
)
    Rother  = CartesianIndices(dst)
    Rreduce = _mapreduce_reduce_indices(src, dst)

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

# Index helpers. Both decode a linear index into a byte-offset by walking compile-time-sized
# segment tuples. The single-segment case (the common one after canonicalization) needs no
# division; multi-segment falls back to a per-element decode (rare: non-adjacent dim sets).

@inline function _outer_decode(tid, outer_strides, outer_sizes)
    isempty(outer_sizes) && return 0
    length(outer_sizes) == 1 && return tid * outer_strides[1]
    base = 0
    tmp  = tid
    @inbounds for i in 1:length(outer_sizes)
        q     = tmp ÷ outer_sizes[i]
        r     = tmp - q * outer_sizes[i]
        base += r * outer_strides[i]
        tmp   = q
    end
    base
end

@inline function _reduce_offset(j, reduce_strides, reduce_sizes)
    isempty(reduce_sizes) && return 0
    # Single segment: j < reduce_size, so the decode is just j * stride — no division.
    length(reduce_sizes) == 1 && return j * reduce_strides[1]
    off = 0
    tmp = j
    @inbounds for i in 1:length(reduce_sizes) - 1
        sz = reduce_sizes[i]
        if ispow2(sz)
            shift = trailing_zeros(sz)
            r   = tmp & (sz - 1)
            tmp = tmp >> shift
        else
            q   = tmp ÷ sz
            r   = tmp - q * sz
            tmp = q
        end
        off += r * reduce_strides[i]
    end
    off += tmp * reduce_strides[end]
    off
end

# GPU kernel: generic fallback — one thread per output element, reducing sequentially
# over the reduced extents using Cartesian indexing. Makes no assumption about the
# source's memory layout, so it handles strided views, adjoints, permuted dims, and
# broadcasts over them. Mirrors the CPU `_mapreduce_nd_cpu_sections!` path.
#
# NOTE: the source is deliberately NOT marked `@Const` here. `@Const` prevents the
# `@inbounds` getindex of some wrappers (e.g. PermutedDimsArray's `genperm`) from
# being elided, leaving a `throw` that the GPU backends cannot compile.
@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_generic!(
    src, dst,
    f, op, init,
    Rother, Rreduce, output_size,
)
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < output_size
        Iother = Rother[tid + 0x1]
        res = init
        for Ireduce in Rreduce
            J = max(Iother, Ireduce)
            res = op(res, f(src[J]))
        end
        dst[Iother] = res
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread_tiled_strided!(
    @Const(src), dst,
    f, op, init, neutral,
    base_offset, reduce_stride,
    output_size, reduce_size,
    ::Val{rows},
) where {rows}
    @uniform N = @groupsize()[1]
    @uniform reduce_threads = N ÷ rows
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Int(...) keeps the index decode signed; a bare `unsigned ÷ unsigned` result feeds
    # the signed index math and adds an Int-conversion throw guard NVPTX keeps (CUDA ~20%).
    row  = Int(unsigned(ithread) % unsigned(rows))
    lane = Int(unsigned(ithread) ÷ unsigned(rows))
    iout = iblock * rows + row

    acc = neutral
    if iout < output_size
        # Contiguous outputs (outer_strides == (1,)), so the source base is just iout;
        # base_offset (0 for a dense source) is folded in once, outside the reduce loop.
        sbase = base_offset + iout
        j = lane
        while j < reduce_size
            acc = op(acc, f(src[sbase + j * reduce_stride + 0x1]))
            j += reduce_threads
        end
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    step = reduce_threads ÷ 2
    while step >= 1
        if lane < step
            dst_lane = row + lane * rows
            src_lane = row + (lane + step) * rows
            sdata[dst_lane + 0x1] = op(sdata[dst_lane + 0x1], sdata[src_lane + 0x1])
        end
        @synchronize()
        step ÷= 2
    end

    if lane == 0x0 && iout < output_size
        dst[iout + 0x1] = op(init, sdata[row + 0x1])
    end
end

# GPU kernel: by_thread — one thread per output element, reducing sequentially.
# Used when there are more output elements than elements in the reduced dimension(s),
# e.g. reduce(+, rand(3, 1000), dims=1) — only 3 elements to reduce per output.

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_thread!(
    @Const(src), dst,
    f, op, init,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size,
)
    # NOTE: index calculations use zero-indexing (fewer ops, matches the CUDA / ROCm / oneAPI /
    # Metal code this is transpiled to), converting to one-indexing only at memory accesses.
    # `base_offset` (the source's element offset within its dense buffer; 0 for a dense
    # source) is folded into the per-output base, so it costs at most one add per thread.
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < output_size
        input_base = base_offset + _outer_decode(tid, outer_strides, outer_sizes)

        res = init
        for j in 0x0:reduce_size - 0x1
            off = _reduce_offset(j, reduce_strides, reduce_sizes)
            res = op(res, f(src[input_base + off + 0x1]))
        end

        dst[tid + 0x1] = res
    end
end

# GPU kernel: by_block — each block reduces one output element, then grid-strides to
# the next output (iout += num_blocks) until all outputs are covered. When
# num_blocks >= output_size, each block handles at most one output (the original,
# non-striding behavior) at zero extra cost — the while loop runs once.
#
# Used for GS_DST_CUTOFF <= dst_size < reduce_size (grid-stride, num_blocks ==
# min(dst_size, TARGET_BLOCKS) < dst_size in general), and also for the
# reduce_groups==1 fallback inside the multigroup branch (num_blocks == dst_size,
# so the loop runs exactly once per block — identical to the pre-grid-stride kernel).

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_by_block!(
    @Const(src), dst,
    f, op, init, neutral,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    output_size, reduce_size, num_blocks,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    iout = iblock
    while true
        input_base = base_offset + _outer_decode(iout, outer_strides, outer_sizes)

        acc = neutral
        j   = ithread
        while j < reduce_size
            off = _reduce_offset(j, reduce_strides, reduce_sizes)
            acc = op(acc, f(src[input_base + off + 0x1]))
            j  += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iout + 0x1] = op(init, sdata[0x1])
        end

        next_iout = iout + num_blocks
        next_iout >= output_size && break

        @synchronize()
        iout = next_iout
    end
end

# GPU kernel: multi-group first pass — several blocks per output element. Block `iblock`
# handles output `iout` and group `igroup`, reducing its interleaved slice of the reduction
# into partial[iout, igroup]. The second pass combines the groups.

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_nd_multigroup!(
    @Const(src), partial,
    f, op, neutral,
    base_offset,
    outer_strides, outer_sizes,
    reduce_strides, reduce_sizes,
    ::Val{output_size}, reduce_size, reduce_groups,
) where {output_size}
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(partial) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Int(...) keeps the index decode signed (see tiled kernel): avoids a CUDA throw guard.
    iout   = Int(unsigned(iblock) % unsigned(output_size))
    igroup = Int(unsigned(iblock) ÷ unsigned(output_size))

    input_base = base_offset + _outer_decode(iout, outer_strides, outer_sizes)

    acc = neutral
    j   = ithread + igroup * N
    while j < reduce_size
        off = _reduce_offset(j, reduce_strides, reduce_sizes)
        acc = op(acc, f(src[input_base + off + 0x1]))
        j  += N * reduce_groups
    end

    sdata[ithread + 0x1] = acc
    @synchronize()

    @inline reduce_group!(@context, op, sdata, N, ithread)

    if ithread == 0x0
        partial[iout + igroup * output_size + 0x1] = sdata[0x1]
    end
end

# GPU kernel: multi-group second pass — one block per output reduces over the `reduce_groups`
# partials and folds in `init`. Launched with a block size sized to `reduce_groups`
# (see _pass2_block_size), avoiding an oversubscribed tree-reduce when reduce_groups
# is much smaller than 256.

@kernel inbounds=true cpu=false unsafe_indices=true function _mapreduce_partial_to_dst!(
    @Const(partial), dst,
    op, init, neutral,
    output_size, reduce_groups,
)
    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    if iblock < output_size
        acc = neutral
        g = ithread
        while g < reduce_groups
            acc = op(acc, partial[iblock + g * output_size + 0x1])
            g += N
        end

        sdata[ithread + 0x1] = acc
        @synchronize()

        @inline reduce_group!(@context, op, sdata, N, ithread)

        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end
