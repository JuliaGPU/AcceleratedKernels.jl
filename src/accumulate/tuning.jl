# Selection and tuning of scan algorithms; the steps are those of sorting (src/sort/tuning.jl).


"""
    ScanPrefixes(; block_size=nothing, items_per_thread=nothing)

GPU scan of a whole array: each block scans a tile of `block_size * items_per_thread` elements
(`block_size` a power of two up to 1024), then the tiles' totals are scanned and added to later
tiles. No block waits for another, so it runs on every backend. The default `items_per_thread` is
at most 8, fewer for wide element types, to keep the tile in local memory.
"""
Base.@kwdef struct ScanPrefixes <: ScanAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end

"""
    DecoupledLookback(; block_size=nothing, items_per_thread=nothing)

GPU scan of a whole array in which each block looks back at earlier blocks' published prefixes
instead of waiting for a separate pass over the tiles' totals; the tiles are those of
[`ScanPrefixes`](@ref). It needs device-scope memory ordering and forward progress between
blocks, so it is rejected on backends that do not guarantee them (currently all but CUDA and
AMDGPU). Elements keep their order, but how they are grouped depends on which blocks have
finished, so floating-point results can differ between runs.
"""
Base.@kwdef struct DecoupledLookback <: ScanAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end

"""
    SliceScan(; block_size=nothing, items_per_thread=nothing)

GPU scan of each slice along `dims`, the algorithm for scans along `dims`. Which threads scan a
slice depends on the layout: one thread per slice for short slices, and where neighbouring
threads then load neighbouring elements (when the dimensions before `dims` hold several
elements); otherwise a block per slice, scanning tiles of `block_size * items_per_thread` consecutive
elements of it (`block_size` a power of two up to 1024, smaller for short slices). When that
leaves too few threads to fill the device, each slice is split into chunks, which takes two more
passes: one over the source, to reduce each chunk, and a scan of the chunks' totals.
"""
Base.@kwdef struct SliceScan <: ScanAlgorithm
    block_size::Union{Nothing, Int} = nothing
    items_per_thread::Union{Nothing, Int} = nothing
end


"""
    ScanTuning(; kwargs...)

Values that drive `Auto` selection and fill unset algorithm fields for scans on one device, as
returned by [`scan_tuning`](@ref).

- `prefer_lookback`: `Auto` picks `DecoupledLookback` for whole-array scans where the backend
  supports it (`_supports_lookback`), else `ScanPrefixes`.
- `block_size`: the block size of every scan kernel algorithm.
- `local_mem_bytes`, `max_items`: the default `items_per_thread` of the kernel algorithms is the
  largest that keeps a tile of the element type within `local_mem_bytes`, and at most
  `max_items`. It is derived from the effective `block_size`, so an explicit `block_size` gets a
  matching default.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.Partitioned`; at least 2.
- `target_blocks`: the blocks a `SliceScan` aims to launch, splitting its slices into chunks when
  they would take fewer.
- `slice_serial_len`: `SliceScan` scans slices of at most this many elements with one thread each.
- `slice_column_inner`: `SliceScan` scans with one thread per slice when the dimensions before
  `dims` hold at least this many elements, so that neighbouring threads load neighbouring
  elements. (Contiguous slices short enough for a block's worth to fit `local_mem_bytes` are
  scanned one per thread from local memory.)
- `slice_min_chunk`: the fewest elements per thread of a `SliceScan` that splits its slices.
- `slice_columns_len`: in arrays too small to split, `SliceScan` scans slices of at most this
  many elements with one thread each (as for `slice_column_inner`), and longer ones a block each.
- `slice_split_min`: `SliceScan` splits the slices it scans with one thread each only in arrays
  of at least this many elements.
- `slice_split_tiles`: `SliceScan` splits slices it scans with a block each when there are too
  few to fill the device and a block would scan more than this many tiles.

The `SliceScan` settings were measured on CUDA (RTX 5080 and 2070); the other defaults reproduce
AK's historical settings. Internal: the fields may change in any release.
"""
Base.@kwdef struct ScanTuning
    prefer_lookback::Bool = false
    block_size::Int = 256
    local_mem_bytes::Int = LOCAL_MEMORY_BUDGET
    max_items::Int = 8
    threads_min_elems::Int = 2
    target_blocks::Int = 256
    slice_serial_len::Int = 8
    slice_column_inner::Int = 4
    slice_min_chunk::Int = 16
    slice_columns_len::Int = 64
    slice_split_min::Int = 1 << 16
    slice_split_tiles::Int = 4
end

"""
    scan_tuning(backend, T) -> ScanTuning

The scan tuning for element type `T` (the running-value type the scan computes in; see
[`accumulate!`](@ref)) on `backend`'s current device; see [`sort_tuning`](@ref) for the
conventions.
"""
scan_tuning(::Backend, ::Type) = ScanTuning()


"""
    _resolve_scan(alg, backend, T, dims, S=T) -> Algorithm

Resolve `alg` for a scan of element type `T` along `dims` (`nothing` for the whole array in
linear order) on `backend`, whose kernels hold partial results of type `S` in local memory (a
lane type when the operator has no known neutral element). Returns `ScanPrefixes`,
`DecoupledLookback`, `SliceScan` or `CPUThreads.Partitioned` with every field set, or throws an
`ArgumentError`.
"""
function _resolve_scan(alg::Algorithm, backend::Backend, ::Type{T}, dims,
                       ::Type{S}=T) where {T, S}
    _checkdomain(alg)
    t = scan_tuning(backend, T)
    a = alg isa Auto ? _select_scan(backend, t, dims) : alg
    a = _fill(a, t, S)
    _check_scan(a, backend, S, dims)
    return a
end

function _select_scan(backend, t::ScanTuning, dims)
    _runs_threads(backend) && return CPUThreads.Partitioned()
    dims === nothing || return SliceScan()
    return _supports_lookback(backend) && t.prefer_lookback ? DecoupledLookback() : ScanPrefixes()
end

function _checkdomain(a::Union{ScanPrefixes, DecoupledLookback, SliceScan})
    _check_pow2(a, :block_size)
    a.block_size === nothing || a.block_size <= 1024 || throw(ArgumentError(
        "$(_algname(a)): `block_size` must be at most 1024, got $(a.block_size)"))
    _check_positive(a, :items_per_thread)
    nothing
end

# The largest number of items per thread whose tile fits the tuning's local-memory budget. Types
# without a definite size and block sizes that `_check_scan` rejects get a placeholder.
function _scan_items(t::ScanTuning, block_size::Int, ::Type{T}) where {T}
    1 <= block_size <= 1024 || return 1
    clamp(t.local_mem_bytes ÷ (block_size * (isbitstype(T) ? max(sizeof(T), 1) : 1)) - 1,
          1, max(t.max_items, 1))
end

function _fill(a::A, t::ScanTuning, ::Type{T}) where {A <: Union{ScanPrefixes, DecoupledLookback, SliceScan}, T}
    block_size = something(a.block_size, t.block_size)
    items_per_thread = isnothing(a.items_per_thread) ? _scan_items(t, block_size, T) :
                                                       a.items_per_thread
    return A(block_size, items_per_thread)
end
_fill(a::CPUThreads.Partitioned, t::ScanTuning, T) = _fill_threads(a, t)
_fill(a::Algorithm, t::ScanTuning, T) =
    throw(ArgumentError("$(_algname(a)) is not a scan algorithm"))

function _check_scan(a::Union{ScanPrefixes, DecoupledLookback}, backend, ::Type{T}, dims) where {T}
    _checkdomain(a)
    _require_kernels(a, backend)
    _check_scan_eltype(a, T)
    dims === nothing || throw(ArgumentError(
        "$(_algname(a)) scans whole arrays; scan along `dims` with `SliceScan` or `Auto()`"))
    if a isa DecoupledLookback && !_supports_lookback(backend)
        throw(ArgumentError(
            "DecoupledLookback needs device-scope memory ordering and forward progress between " *
            "blocks, which $(_backend_name(backend)) does not guarantee; use `ScanPrefixes`"))
    end
    # Keep the kernels' index arithmetic far from overflow
    widemul(a.block_size, a.items_per_thread) <= typemax(Int32) || throw(ArgumentError(
        "$(_algname(a)): `block_size * items_per_thread` must be at most $(typemax(Int32))"))
    nothing
end

function _check_scan(a::SliceScan, backend, ::Type{T}, dims) where {T}
    _checkdomain(a)
    _require_kernels(a, backend)
    _check_scan_eltype(a, T)
    dims === nothing && throw(ArgumentError(
        "SliceScan scans along `dims`; scan whole arrays with `ScanPrefixes` or `Auto()`"))
    widemul(a.block_size, a.items_per_thread) <= typemax(Int32) || throw(ArgumentError(
        "SliceScan: `block_size * items_per_thread` must be at most $(typemax(Int32))"))
    nothing
end

function _check_scan(a::CPUThreads.Partitioned, backend, T, dims)
    _checkdomain(a)
    _require_threads(a, backend)
    # Each task's part must hold two elements for the exclusive scan's carries
    a.min_elems >= 2 || throw(ArgumentError(
        "CPUThreads.Partitioned: scans need `min_elems` of at least 2, got $(a.min_elems)"))
    nothing
end

_check_scan_eltype(a, ::Type{T}) where {T} = isbitstype(T) || throw(ArgumentError(
    "$(_algname(a)): the element type $T is not a bits type, which the kernels need"))
