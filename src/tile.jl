# Tiling primitive ported from GemmKernels.jl. Shape lives in the `size` type parameter (loops unroll,
# `@localmem` sizing is static), position in the `base`/`offset` fields. Move tiles with
# `translate_base`/`translate_offset` and read the flat coordinate off `tile.index`; bounds are the
# consumer's job. Only the 1-D core is ported; the 2-D iterator machinery can be added when needed.
module Tiling

export Tile, translate_base, translate_offset

struct Tile{size, names, T}
    base::NamedTuple{names, T}
    offset::NamedTuple{names, T}
end

Tile(; kw_args...) = Tile((; kw_args...))

@inline Tile(size::NamedTuple{names, T}) where {names, T} =
    Tile{size, names, T}(map(x -> 0, size), map(x -> 0, size))

@generated function _getproperty_impl(tile::Tile{size, names, T}, ::Val{sym}) where {names, T, sym, size}
    if sym === :base || sym === :offset
        return :(getfield(tile, sym))
    elseif sym === :size
        return size
    elseif sym === :index
        return :(map(+, getfield(tile, :base), getfield(tile, :offset)))
    else
        syms = ntuple(i -> Symbol(String(sym)[i]), length(String(sym)))
        return :(_projection_impl(NamedTuple{$syms}(getfield(tile, :base)),
                                  NamedTuple{$syms}(getfield(tile, :offset)),
                                  NamedTuple{$syms}(size)))
    end
end

@inline Base.getproperty(tile::Tile{size, names, T}, sym::Symbol) where {names, T, size} =
    _getproperty_impl(tile, Val(sym))

@inline _projection_impl(base::NamedTuple{names, T}, offset::NamedTuple{names, T},
                         size::NamedTuple{names, T}) where {names, T} =
    Tile{size, names, T}(base, offset)

@inline function translate_base(tile::Tile{size, names, T}, offset::NamedTuple{names, T}) where {names, T, size}
    return Tile{size, names, T}(map(+, tile.base, offset), tile.offset)
end
@inline translate_base(tile::Tile{size, names, T}, offset::Tuple) where {names, T, size} =
    translate_base(tile, NamedTuple{names}(offset))

@inline function translate_offset(tile::Tile{size, names, T}, offset::NamedTuple{names, T}) where {names, T, size}
    return Tile{size, names, T}(tile.base, map(+, tile.offset, offset))
end
@inline translate_offset(tile::Tile{size, names, T}, offset::Tuple) where {names, T, size} =
    translate_offset(tile, NamedTuple{names}(offset))

end # module Tiling
