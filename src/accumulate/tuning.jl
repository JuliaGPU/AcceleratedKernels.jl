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
    SliceScan(; block_size=nothing)

GPU scan of each slice along `dims`: one thread per slice when there are more slices than
elements per slice, otherwise one block (`block_size` a power of two up to 1024) per slice.
The algorithm for scans along `dims`.
"""
Base.@kwdef struct SliceScan <: ScanAlgorithm
    block_size::Union{Nothing, Int} = nothing
end


"""
    ScanTuning(; kwargs...)

Values that drive `Auto` selection and fill unset algorithm fields for scans on one device, as
returned by [`scan_tuning`](@ref).

- `prefer_lookback`: `Auto` picks `DecoupledLookback` for whole-array scans where the backend
  supports it (`_supports_lookback`), else `ScanPrefixes`.
- `block_size`: the block size of every scan kernel algorithm.
- `local_mem_bytes`, `max_items`: the default `items_per_thread` of `ScanPrefixes` and
  `DecoupledLookback` is the largest that keeps a tile of the element type within
  `local_mem_bytes`, and at most `max_items`. It is derived from the effective `block_size`, so an
  explicit `block_size` gets a matching default.
- `threads_min_elems`: the default `min_elems` of `CPUThreads.Partitioned`; at least 2.

The defaults reproduce AK's historical settings. Internal: the fields may change in any release.
"""
Base.@kwdef struct ScanTuning
    prefer_lookback::Bool = false
    block_size::Int = 256
    local_mem_bytes::Int = LOCAL_MEMORY_BUDGET
    max_items::Int = 8
    threads_min_elems::Int = 2
end

"""
    scan_tuning(backend, T) -> ScanTuning

The scan tuning for element type `T` (the destination's) on `backend`'s current device; see
[`sort_tuning`](@ref) for the conventions.
"""
scan_tuning(::Backend, ::Type) = ScanTuning()


"""
    _resolve_scan(alg, backend, T, dims) -> Algorithm

Resolve `alg` for a scan of element type `T` along `dims` (`nothing` for the whole array in
linear order) on `backend`. Returns `ScanPrefixes`, `DecoupledLookback`, `SliceScan` or
`CPUThreads.Partitioned` with every field set, or throws an `ArgumentError`.
"""
function _resolve_scan(alg::Algorithm, backend::Backend, ::Type{T}, dims) where {T}
    _checkdomain(alg)
    t = scan_tuning(backend, T)
    a = alg isa Auto ? _select_scan(backend, t, dims) : alg
    a = _fill(a, t, T)
    _check_scan(a, backend, T, dims)
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
    a isa SliceScan || _check_positive(a, :items_per_thread)
    nothing
end

# The largest number of items per thread whose tile fits the tuning's local-memory budget. Types
# without a definite size and block sizes that `_check_scan` rejects get a placeholder.
function _scan_items(t::ScanTuning, block_size::Int, ::Type{T}) where {T}
    1 <= block_size <= 1024 || return 1
    clamp(t.local_mem_bytes ÷ (block_size * (isbitstype(T) ? max(sizeof(T), 1) : 1)) - 1,
          1, max(t.max_items, 1))
end

function _fill(a::A, t::ScanTuning, ::Type{T}) where {A <: Union{ScanPrefixes, DecoupledLookback}, T}
    block_size = something(a.block_size, t.block_size)
    items_per_thread = isnothing(a.items_per_thread) ? _scan_items(t, block_size, T) :
                                                       a.items_per_thread
    return A(block_size, items_per_thread)
end
_fill(a::SliceScan, t::ScanTuning, T) = SliceScan(something(a.block_size, t.block_size))
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
