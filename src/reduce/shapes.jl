# The launch shapes of a reduction along `dims`: which kernel reduces it, with how many threads
# or blocks per output, and the scratch that takes.

# Fewer outputs than this split their reductions across blocks (`ByBlocks`), as a block per
# output leaves most of a GPU idle (5 or 9 blocks on an 84-SM one); `ByLanes` needs at least this
# many
const FEW_OUTPUTS = 32

# The fewest elements each thread reduces in a reduction split across blocks, which bounds the
# blocks per output
const SPLIT_MIN_ITEMS = 8

# The outputs per block of the tiled kernel (`Tiled`): 32 lanes each in a block of 256
const TILED_OUTPUTS = 8

# The most consecutive outputs per block of the columns kernel (fewer outputs take the next power
# of two), so that a warp's loads of one step are consecutive elements; and the loads each of its
# threads issues together
const COLUMNS_OUTPUTS = 32
const COLUMNS_ITEMS_PER_THREAD = 8
# The `ByLanes` reduction: at most this many lanes per output (the kernel is compiled for each
# number of lanes), doubling the lanes of runs with few outputs up to this many threads
const MAX_LANES = 32
const LANES_MIN_THREADS = 8192

# The fewest outputs the columns kernel is used for: with 4 or fewer, its blocks' scattered loads
# lose to `ByBlocks` on Metal and oneAPI
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


# The launch shapes, after CUB's `DeviceSegmentedReduce` (a thread, a group of lanes or a block
# per output) and PyTorch's `ReduceConfig` (tiles of contiguous outputs, and several blocks per
# output). Each but `Whole` has a `launch!` method (mapreduce_nd.jl).
abstract type ReduceShape end

# One thread per output, which reduces its slice in order
struct ByThread <: ReduceShape end

# `lanes` threads per output, which read a run of consecutive elements in turn, several outputs
# per block
struct ByLanes <: ReduceShape
    lanes::Int
end

# A block per output: `blocks` blocks, which reduce the outputs `iblock`, `iblock + blocks`, ...
struct ByBlock <: ReduceShape
    blocks::Int
end

# `groups` blocks per output, whose partial results a second pass combines
struct ByBlocks <: ReduceShape
    groups::Int
end

# Tiles of `outputs` contiguous outputs per block, whose elements are a stride apart, each output's
# reduction split across `groups` blocks; `ordered` combines the elements in their order
struct Columns <: ReduceShape
    outputs::Int
    groups::Int
    ordered::Bool
end

# The tiled kernel: TILED_OUTPUTS contiguous outputs per block, in one pass
struct Tiled <: ReduceShape end

# In element order: `groups` blocks per output, combining neighbouring partial results
struct Ordered <: ReduceShape
    groups::Int
end

# A single output of a contiguous source: the whole-array kernels (`mapreduce_1d_gpu`, which
# `mapreduce_nd!` calls itself), in element order if `ordered`
struct Whole <: ReduceShape
    ordered::Bool
end

# The blocks per output, whose partial results take scratch where there are several
split_groups(shape) = shape isa Union{ByBlocks, Columns, Ordered} ? shape.groups : 1


# The launch shape of a strided reduction along `dims` into `dst_size` outputs of `reduce_size`
# elements each (merged into the segments `segs`). In order:
#   - outputs in runs of contiguous ones, each reducing elements a stride apart: see
#     `strided_shape`, where it takes them;
#   - at least FEW_OUTPUTS outputs, each reducing a run of consecutive elements of a length the
#     tuning's `lanes_*` settings allow: `ByLanes`;
#   - at least as many outputs as elements each: `ByThread`, or `ByBlock` where that reads
#     with a stride or fills too few blocks (see below);
#   - fewer than FEW_OUTPUTS outputs: `ByBlocks`, its blocks per output launching about the
#     tuning's `target_blocks` blocks, or `ByBlock` with a block per output where that is one;
#   - else `ByBlock`, grid-striding over the outputs with at most `target_blocks` blocks.
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
       (t.lanes_short_runs || reduce_size >= block_size)
        lanes = lanes_per_output(dst_size, reduce_size, block_size)
        if lanes > 1 && cld(reduce_size, lanes) * accsize(A) <= t.lanes_max_bytes
            return ByLanes(lanes)
        end
    end

    # A block per output rather than a thread, for slices of at least a block of elements: where
    # each slice is consecutive elements, which a thread per output would read with a stride
    # (and the lanes did not take), and for as many outputs as elements each that would fill
    # too few blocks (more outputs read faster with a thread each, coalesced across outputs)
    use_by_block_for_coalescing =
        dst_size >= reduce_size && reduce_size >= block_size &&
        single_reduce && rstrides[1] == 1
    use_by_block_for_low_occupancy =
        dst_size == reduce_size && reduce_size >= block_size &&
        cld(dst_size, block_size) < target_blocks && single_reduce

    use_by_block = use_by_block_for_coalescing || use_by_block_for_low_occupancy

    if dst_size >= reduce_size && !use_by_block
        return ByThread()
    elseif !use_by_block && dst_size < FEW_OUTPUTS
        groups = min(
            cld(reduce_size, block_size),
            block_size,
            cld(target_blocks, dst_size),
            cld(reduce_size, block_size * SPLIT_MIN_ITEMS),
        )
        return groups > 1 ? ByBlocks(groups) : ByBlock(dst_size)
    else
        return ByBlock(min(dst_size, target_blocks))
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
#     second pass, which only reductions of at least the tuning's `columns_split_min_elems`
#     elements afford.
# Tiles of 32 outputs that need splitting take longer than the tiled kernel where its 8 outputs
# per block launch enough blocks, from 1.7 times as long on an RTX 5080 for a 1024×1024 matrix to
# 2.5 times on an Arc A770; with more outputs, the columns kernel's several loads in flight beat
# its single one (a 8192×8192 Float32 matrix takes 296 instead of 416 µs on the RTX 5080).
function strided_shape(dst_size, reduce_size, run, block_size, t::ReduceTuning,
                       ::Type{A}) where {A}
    if widemul(dst_size, accsize(A)) >= BY_THREAD_ROW_BYTES || reduce_size < COLUMNS_MIN_REDUCE
        return ByThread()
    end
    threads = min(block_size, COLUMNS_MAX_THREADS)
    outputs = columns_outputs(run)
    runs = dst_size ÷ run
    tiles = runs * cld(run, outputs)
    rows = TILED_OUTPUTS
    if runs == 1 && dst_size >= reduce_size && tiles < t.target_blocks &&
       block_size % rows == 0 && ispow2(block_size ÷ rows) && reduce_size >= block_size ÷ rows
        return Tiled()
    end
    run >= COLUMNS_MIN_OUTPUTS && threads >= outputs || return nothing
    lanes = threads ÷ outputs
    groups = clamp(cld(t.target_blocks, tiles), 1, cld(reduce_size, lanes * SPLIT_MIN_ITEMS))
    groups > 1 && widemul(dst_size, reduce_size) < t.columns_split_min_elems && return nothing
    return Columns(outputs, groups, false)
end

# The launch shape of a reduction in element order (`commutative=false`), from the shape
# `reduce_shape` chose for it: one thread per output, which reduces its slice in order,
# where that shape is one, or for runs shorter than a block; the columns kernel in order for
# strided reductions into contiguous outputs, which it reads coalesced across outputs; else
# `Ordered`, a block per output, or the groups of a `ByBlocks` reduction.
function ordered_shape(shape::ReduceShape, dst_size, reduce_size, block_size)
    (shape isa ByThread || shape isa ByLanes && reduce_size < block_size) && return ByThread()
    shape isa Columns && return Columns(shape.outputs, shape.groups, true)
    if shape isa Tiled
        outputs = columns_outputs(dst_size)
        min(block_size, COLUMNS_MAX_THREADS) >= outputs && return Columns(outputs, 1, true)
    end
    return Ordered(shape isa ByBlocks ? shape.groups : 1)
end

# The lanes per output of a `ByLanes` reduction: about the square root of the length of the
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

# The size of the accumulator, which the lanes of a `ByLanes` reduction each reduce a share of
# the run into; an indicative size for types without one
accsize(::Type{A}) where {A} = isbitstype(A) && sizeof(A) > 0 ? sizeof(A) : 8

# The outputs per block of the columns kernel
columns_outputs(dst_size) = min(nextpow(2, dst_size), COLUMNS_OUTPUTS)

# The launch of a reduction along `dims` of `src` into `dst_size` outputs, chosen once by its
# setup; the threaded algorithm takes none. `shape` is that of `reduce_shape`, or of
# `ordered_shape` for `commutative=false`, or `Whole` for a single output of a contiguous source;
# `nothing` where no kernel runs (at most one element per output, no outputs, or no accumulator
# type). `buffer` is where the kernels find the source's elements (see `buffer_kind`), from
# `offset` on, and `segs` the segments of its layout (see `strided_layout`, `linear_layout` and
# `merge_dims`).
struct DimsLaunch{G}
    # (abstract, so that the setups holding it infer; the launch dispatches on it)
    shape::Union{Nothing, ReduceShape}
    buffer::Symbol
    offset::Int
    segs::G
end

select_launch(src, backend, alg, ::Type{A}, rdims, dst_size, commutative) where {A} = nothing
function select_launch(src, backend, alg::BlockReduce, ::Type{A}, rdims, dst_size,
                       commutative) where {A}
    strided = strided_layout(src)
    layout = isnothing(strided) ? linear_layout(src) : strided
    segs = merge_dims(size(src), layout.strides, rdims)
    kind, offset = buffer_kind(src, strided), layout.offset::Int
    launch(shape) = DimsLaunch(shape, kind, offset, segs)
    len = Base.prod(size(src)[d] for d in rdims; init=1)
    (len <= 1 || dst_size == 0 || A === Union{}) && return launch(nothing)
    (rstrides, _, nr), _ = segs
    if dst_size == 1 && (isnothing(strided) || nr == 1 && rstrides[1] == 1)
        return launch(Whole(!commutative))
    end
    shape = reduce_shape(dst_size, len, segs, alg.block_size, reduce_tuning(backend, A), A)
    commutative || (shape = ordered_shape(shape, dst_size, len, alg.block_size))
    return launch(shape)
end

# The scratch of the launch `launch` of a reduction along `dims` of `src` into `dst_size`
# outputs: the partial results of a reduction in two passes, with the type of the partial-result
# seed `seed` (see `reduce_seed`)
launch_scratch(launch, src, backend, alg, A, seed, rdims, dst_size) = (;)
function launch_scratch(launch::DimsLaunch, src, backend, alg, ::Type{A}, seed, rdims,
                        dst_size) where {A}
    shape = launch.shape
    if shape isa Whole
        len = Base.prod(size(src)[d] for d in rdims; init=1)
        n = mapreduce_1d_partials(len, alg.block_size, alg.items_per_thread,
                                  reduce_tuning(backend, A).target_blocks)
        return n == 0 ? (;) : (; partials=scratch(typeof(seed), n))
    end
    groups = split_groups(shape)
    return groups > 1 ? (; partials=scratch(typeof(seed), dst_size, groups)) : (;)
end
