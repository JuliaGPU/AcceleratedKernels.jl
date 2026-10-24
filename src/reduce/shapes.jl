# The launch shapes of a reduction along `dims`: which kernel reduces it, with how many threads
# or blocks per output, and the scratch that takes.

# Below this many output elements, splitting a single output's reduction across multiple
# blocks (multigroup) is preferred over grid-striding by_block, because grid-stride with
# too few blocks cannot fill the GPU on its own. At or above this, by_block grid-strides.
const FEW_OUTPUTS = 32

# Minimum reduction-loop iterations per thread in the multigroup first pass.
# Caps reduce_groups so splitting a reduction doesn't shrink per-thread work
# below the point where launch/scheduling overhead dominates actual work.
const SPLIT_MIN_ITEMS = 8

# Number of contiguous output rows handled by each tiled-strided block. With the default
# 256-thread block this gives 32 lanes per row.
const TILED_OUTPUTS = 8

# The most consecutive outputs per block of the columns kernel (fewer outputs take the next power
# of two), so that a warp's loads of one step are consecutive elements; and the loads each of its
# threads issues together
const COLUMNS_OUTPUTS = 32
const COLUMNS_ITEMS_PER_THREAD = 8
# The `:by_lanes` reduction: at most this many lanes per output (the kernel is compiled for each
# number of lanes), doubling the lanes of runs with few outputs up to this many threads
const MAX_LANES = 32
const LANES_MIN_THREADS = 8192

# The fewest outputs the columns kernel is used for: with 4 or fewer, its blocks' scattered loads
# lose to `:multigroup` on Metal and oneAPI
const COLUMNS_MIN_OUTPUTS = 8
# The most threads per block of the columns kernel: its unrolled loads take more registers than
# the other kernels, and Metal allows it fewer than 1024 threads per block
const COLUMNS_MAX_THREADS = 256
# A strided reduction into contiguous outputs takes one thread per output when a row of outputs,
# which those threads load together at each step, is at least this many bytes, or when the
# reduction is shorter than COLUMNS_MIN_REDUCE. Measured on an RTX 5080 and an RTX 6000 Ada: one
# thread per output is as fast as the other kernels from 64 KiB of Float32 or BFloat16 outputs,
# and 2.3 (Float32) to 4.6 (BFloat16) times slower with 4096 outputs, which fill only 16 blocks.
# (Devices with a few compute units fill sooner: on a Ryzen 9950X's integrated GPU and a Jetson
# Xavier, 2048 outputs of 768 elements reduce 1.4 times as fast with one thread each.)
const BY_THREAD_ROW_BYTES = 64 * 1024
const COLUMNS_MIN_REDUCE = 32


# The launch shape of a strided reduction along `dims`, and the number of blocks per output of a
# `:multigroup` or `:columns` reduction, or of threads per output of a `:by_lanes` one:
#   - at least FEW_OUTPUTS outputs, each reducing a
#     run of consecutive elements, of a length the
#     tuning's `segments_*` settings allow             -> :by_lanes (a group of threads per
#                                                          output, several outputs per block)
#   - runs of contiguous outputs, each reducing one
#     strided segment                                  -> :by_thread, :tiled_strided or :columns,
#                                                          see `strided_shape`
#   - dst_size >= reduce_size                          -> :by_thread
#   - dst_size == 1, or dst_size < FEW_OUTPUTS
#     (and dst_size < reduce_size)                     -> :multigroup (split one reduction
#                                                          across blocks, needs a second-pass
#                                                          combine and `partials` scratch), or
#                                                          :by_block_each when that is one block
#   - otherwise (FEW_OUTPUTS <= dst_size < reduce_size) -> :by_block, grid-striding over
#                                                            outputs, single pass
#
# Rationale: grid-striding by_block launches `min(dst_size, target_blocks)` blocks;
# for very small dst_size (e.g. 5 or 9) that under-fills an 84-SM GPU, so splitting
# the (large) reduction itself across many blocks via multigroup is still better.
# For dst_size==1 there is nothing to grid-stride over, so multigroup is the only
# option regardless of FEW_OUTPUTS.
function reduce_shape(dst_size, reduce_size, segs, block_size, t::ReduceTuning,
                      ::Type{A}) where {A}
    target_blocks = t.target_blocks
    (rstrides, rsizes, nr), (ostrides, osizes, no) = segs
    single_reduce = nr == 1 && rsizes[1] != 0

    if no >= 1 && ostrides[1] == 1 && nr == 1 && rstrides[1] > 1
        shape = strided_shape(dst_size, reduce_size, osizes[1], block_size, t, A)
        shape === nothing || return shape
    end

    # Each output reduces a run of consecutive elements: a group of lanes per output reads it
    # with consecutive loads, and a block holds several outputs, so that it stays busy on short
    # runs (where one block per output idles most of its threads, and one thread per output
    # reads with a stride). Runs too short for two lanes keep one thread per output, and runs
    # too long for the tuning's bytes per lane keep a block per output, which reads them with
    # vector loads.
    if single_reduce && rstrides[1] == 1 && dst_size >= FEW_OUTPUTS &&
       (t.segments_short_runs || reduce_size >= block_size)
        lanes = lanes_per_output(dst_size, reduce_size, block_size)
        if lanes > 1 && cld(reduce_size, lanes) * accsize(A) <= t.segments_lane_bytes
            return :by_lanes, lanes
        end
    end

    # by_block override 1: when the reduced dimension is the fastest-varying one
    # (reduce_strides==(1,)), by_thread's per-thread strided access is badly
    # uncoalesced, while by_block lets consecutive threads read consecutive elements.
    use_by_block_for_coalescing =
        dst_size >= reduce_size && reduce_size >= block_size &&
        single_reduce && rstrides[1] == 1

    # by_block override 2: when by_thread would launch too few blocks for a square-like
    # output/reduction shape, split each output reduction across a full block. This
    # fallback is for layouts `strided_shape` does not take. Avoid applying this to
    # wide-output shapes, where by_thread's cross-output coalescing is better than a
    # strided block reduction.
    use_by_block_for_low_occupancy =
        dst_size == reduce_size && reduce_size >= block_size &&
        cld(dst_size, block_size) < target_blocks && single_reduce

    use_by_block = use_by_block_for_coalescing || use_by_block_for_low_occupancy

    if dst_size >= reduce_size && !use_by_block
        return :by_thread, 0
    elseif use_by_block
        return :by_block, 0
    elseif dst_size == 1 || dst_size < FEW_OUTPUTS
        reduce_groups = min(
            cld(reduce_size, block_size),
            block_size,
            cld(target_blocks, dst_size),
            cld(reduce_size, block_size * SPLIT_MIN_ITEMS),
        )
        return reduce_groups > 1 ? (:multigroup, reduce_groups) : (:by_block_each, 0)
    else
        return :by_block, 0
    end
end

# The launch shape of a strided reduction into runs of `run` contiguous outputs (e.g. `dims=2` of a
# matrix, or of a 3-d array, whose third dimension holds the runs), or `nothing` for the shapes of
# `reduce_shape`:
#   - one thread per output for rows of BY_THREAD_ROW_BYTES of outputs, or short reductions;
#   - else, with at least as many outputs as elements in each, and too few for tiles of 32 outputs
#     to launch the tuning's `target_blocks`, the tiled kernel (of a single run): 8 outputs per
#     block, in one pass;
#   - else the columns kernel, with tiles of up to 32 outputs of a run, and each output's
#     reduction split across enough blocks to launch `target_blocks`. A split reduction takes a
#     second pass, which only reductions of at least the tuning's `columns_min_elements` elements
#     afford.
# Tiles of 32 outputs that need splitting take longer than the tiled kernel where its 8 outputs
# per block launch enough blocks, from 1.7 times as long on an RTX 5080 for a 1024×1024 matrix to
# 2.5 times on an Arc A770; with more outputs, the columns kernel's several loads in flight beat
# its single one (a 8192×8192 Float32 matrix takes 296 instead of 416 µs on the RTX 5080).
function strided_shape(dst_size, reduce_size, run, block_size, t::ReduceTuning,
                       ::Type{A}) where {A}
    if widemul(dst_size, accsize(A)) >= BY_THREAD_ROW_BYTES || reduce_size < COLUMNS_MIN_REDUCE
        return :by_thread, 0
    end
    threads = min(block_size, COLUMNS_MAX_THREADS)
    cols = columns_outputs(run)
    runs = dst_size ÷ run
    tiles = runs * cld(run, cols)
    rows = TILED_OUTPUTS
    if runs == 1 && dst_size >= reduce_size && tiles < t.target_blocks &&
       block_size % rows == 0 && ispow2(block_size ÷ rows) && reduce_size >= block_size ÷ rows
        return :tiled_strided, 0
    end
    run >= COLUMNS_MIN_OUTPUTS && threads >= cols || return nothing
    lanes = threads ÷ cols
    groups = clamp(cld(t.target_blocks, tiles), 1, cld(reduce_size, lanes * SPLIT_MIN_ITEMS))
    groups > 1 && widemul(dst_size, reduce_size) < t.columns_min_elements && return nothing
    return :columns, groups
end

# The launch shape of a reduction in element order (`commutative=false`), from the shape
# `reduce_shape` chose for it: one thread per output, which reduces its slice in order,
# where that shape is one, or for runs shorter than a block; the columns kernel in order for
# strided reductions into contiguous outputs, which it reads coalesced across outputs; else
# `:ordered`, a block per output, or the groups of a `:multigroup` reduction. The groups are those
# of the chosen shape, so that partial results take the same scratch.
function ordered_shape(shape, reduce_groups, dst_size, reduce_size, block_size)
    (shape === :by_thread || shape === :by_lanes && reduce_size < block_size) &&
        return :by_thread, 0
    shape === :columns && return :columns_ordered, reduce_groups
    shape === :tiled_strided && min(block_size, COLUMNS_MAX_THREADS) >= columns_outputs(dst_size) &&
        return :columns_ordered, 1
    return :ordered, shape === :multigroup ? reduce_groups : 1
end

# The lanes per output of a `:by_lanes` reduction: about the square root of the length of the
# runs, and more (keeping 16 elements per lane) while fewer than LANES_MIN_THREADS threads
# would run; a power of two of at most 32 and the block size. Measured on an M1, from runs of 8
# elements (2 lanes) to 2048 (32 lanes).
function lanes_per_output(dst_size, reduce_size, block_size)
    max_lanes = min(MAX_LANES, block_size)
    lanes = clamp(prevpow(2, isqrt(reduce_size)), 1, max_lanes)
    while lanes < max_lanes && 32 * lanes <= reduce_size &&
          widemul(dst_size, lanes) < LANES_MIN_THREADS
        lanes *= 2
    end
    return lanes
end

# The size of the accumulator, which the lanes of a `:by_lanes` reduction each reduce a share of
# the run into; an indicative size for types without one
accsize(::Type{A}) where {A} = isbitstype(A) && sizeof(A) > 0 ? sizeof(A) : 8

# The outputs per block of the columns kernel
columns_outputs(dst_size) = min(nextpow(2, dst_size), COLUMNS_OUTPUTS)

# The scratch of a reduction along `dims` into `dst_size` outputs: the partial results of a
# `:multigroup` reduction, of a `:columns` one in two passes or of the whole-array kernels, with
# the type of the partial-result seed `seed` (see `reduce_seed`)
function mapreduce_nd_sizes(src, backend, alg, ::Type{A}, seed, rdims, dst_size) where {A}
    alg isa BlockReduce && A !== Union{} || return (;)
    src_sizes = size(src)
    len = Base.prod(src_sizes[d] for d in rdims; init=1)
    (len <= 1 || dst_size == 0) && return (;)
    if dst_size == 1 && linear_source(src, rdims) !== nothing
        n = mapreduce_1d_partials(len, alg.block_size, alg.items_per_thread,
                                  reduce_tuning(backend, A).target_blocks, 0; device=true)
        return n == 0 ? (;) : (; partials=scratch(typeof(seed), n))
    end
    segs = merge_dims(src_sizes, source_layout(src)[3], rdims)
    shape, reduce_groups = reduce_shape(dst_size, len, segs, alg.block_size,
                                        reduce_tuning(backend, A), A)
    shape === :multigroup || shape === :columns && reduce_groups > 1 || return (;)
    return (; partials=scratch(typeof(seed), dst_size, reduce_groups))
end
