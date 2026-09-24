# The reduction contract's machinery (see `mapreducedim!`): the accumulator type, the value that
# seeds partial results, and how a finished partial becomes an output value.

# The mapped element type of a reduction source. Base's `combine_eltypes` does not see through
# the `Extruded` wrappers of a preprocessed `Broadcasted`, so its element types are derived here.
_mapped_eltype(f, src) = Base.promote_op(f, _source_eltype(src))
_source_eltype(src::AbstractArray) = eltype(src)
_source_eltype(bc::Base.Broadcast.Broadcasted) = Base.promote_op(bc.f, Base.map(_source_eltype, bc.args)...)
_source_eltype(x::Base.Broadcast.Extruded) = eltype(x.x)
_source_eltype(x) = Base.Broadcast._broadcast_getindex_eltype(x)
# A `Broadcasted` source as a dense array of its element type (`materialize` would take the
# element type from `combine_eltypes`, which gives `Any` for `Extruded` arguments)
_materialize_source(bc::Base.Broadcast.Broadcasted) = copyto!(similar(bc, _source_eltype(bc)), bc)

# The type of a reduction's first partial value when no `init` is given.
_first_type(op, ::Type{M}) where {M} = Base.promote_op(Base.reduce_first, Core.Typeof(op), M)

# The accumulator type of a reduction of mapped elements of type `M` into `S` (the type of `init`
# or of the destination, `Union{}` for neither): the types Base's fold `op(op(S, M), M)...` goes
# through, joined with `promote_type`. In a parallel reduction every element is also a
# one-element partial result (`Base.reduce_first(op, x)`), and partial results are combined with
# each other, so those types are joined as well; `S` itself is not a partial result (`init` and
# the destination's values are applied once, at the end). So `sum(Int8[...])` accumulates in
# `Int`, as Base's `add_sum` does, and partial results have one type however many elements they
# combine. `Union{}` when inference shows that the fold always throws; only a reduction with
# elements to combine is an error then (`_check_acctype`).
function _reduce_acctype(op, ::Type{S}, ::Type{M}) where {S, M}
    A = _first_type(op, M)
    if S !== Union{}
        B = Base.promote_op(op, S, M)
        B === Union{} && return Union{}
        A = A === Union{} ? B : promote_type(A, B)
    end
    A === Union{} && return Union{}
    for _ in 1:8
        B = Base.promote_op(op, A, M)
        B === Union{} && return Union{}
        B = promote_type(A, B)
        C = Base.promote_op(op, B, B)
        C === Union{} || (B = promote_type(B, C))
        B == A && return A
        A = B
    end
    return A
end

_check_acctype(op, f, ::Type{A}) where {A} = A === Union{} && throw(ArgumentError(
    "`$op` cannot combine the mapped elements (`$f`): calling it always throws for their types"))

# The accumulator type of a reduction: the caller's `acctype`, else the rule of
# `_reduce_acctype`. An `acctype` is rejected only where the types rule it out, when a value that
# a partial result takes (`Base.reduce_first`'s of an element, or `op`'s of a partial result and
# an element or of two partial results) has no conversion to it at all; whether the values fit is
# the caller's obligation.
_acctype(op, ::Type{S}, ::Type{M}, ::Nothing) where {S, M} = _reduce_acctype(op, S, M)
function _acctype(op, ::Type{S}, ::Type{M}, ::Type{A}) where {S, M, A}
    M === Union{} && return A           # (no elements to hold)
    _check_holds(A, _first_type(op, M), "a one-element partial result (`Base.reduce_first`)")
    for (T, what) in ((Base.promote_op(op, A, M), "a partial result and an element"),
                      (Base.promote_op(op, A, A), "two partial results"))
        T === Union{} && throw(ArgumentError("`acctype=$A`: `$op` of $what always throws"))
        _check_holds(A, T, "`$op` of $what")
    end
    return A
end
_check_holds(::Type{A}, ::Type{T}, what) where {A, T} =
    T === Union{} || Base.promote_op(convert, Type{A}, T) !== Union{} ||
    throw(ArgumentError("`acctype=$A` cannot hold $what, of type `$T`"))
_acctype(op, ::Type, ::Type, acctype) =
    throw(ArgumentError("`acctype` must be a type or `nothing`, got $(repr(acctype))"))

# A partial result of a reduction whose operator has no known neutral element. It starts out
# empty and takes its first value from `Base.mapreduce_first`; combining with an empty lane
# returns the other operand. For associative and commutative operators this gives the same result
# as seeding with a neutral element.
#
# WORKAROUND(Intel NEO): the value comes first, because Intel's OpenCL compiler miscompiles lanes
# whose aggregate value (e.g. `findmin`'s tuple) follows the flag (JuliaGPU/OpenCL.jl#502). The
# order is otherwise arbitrary; nothing to restore once fixed.
#
# An empty lane of a bits type (every lane kernels see) holds zero bits; one of another type leaves
# the value undefined, which is how it is recognised.
struct _Lane{T}
    value::T
    valid::Bool
    _Lane{T}() where {T} = _inline_value(T) ? new{T}(_zero_bits(T), false) : new{T}()
    _Lane{T}(value) where {T} = new{T}(value, true)
    _Lane{T}(value, valid::Bool) where {T} = new{T}(value, valid)
end

_inline_value(::Type{T}) where {T} = isbitstype(T) || Base.isbitsunion(T)
@inline _valid(l::_Lane{T}) where {T} = _inline_value(T) ? l.valid : isdefined(l, :value)

# A value of the bits type or bits union `T` with all-zero bits
function _zero_bits(::Type{T}) where {T}
    isbitstype(T) || return _zero_bits(first(Base.uniontypes(T)))
    Base.issingletontype(T) && return T.instance
    bytes = zeros(UInt8, sizeof(T))
    return GC.@preserve bytes unsafe_load(Ptr{T}(pointer(bytes)))
end

struct _LaneMap{F, OP, T}
    f::F
    op::OP
end
_LaneMap{T}(f::F, op::OP) where {T, F, OP} = _LaneMap{F, OP, T}(f, op)
@inline (m::_LaneMap{F, OP, T})(x) where {F, OP, T} = _Lane{T}(Base.mapreduce_first(m.f, m.op, x))

struct _LaneOp{OP}
    op::OP
end
@inline function (o::_LaneOp)(a::_Lane{T}, b::_Lane{T}) where {T}
    if _inline_value(T)
        # WORKAROUND(LLVM SPIR-V back-end): the value and the flag are chosen separately, because
        # llc crashes ("SPIRV emit intrinsics", LLVM 22.1) on the `select` between whole lanes
        # that the early returns below compile to (llvm/llvm-project#226218). Once fixed, drop
        # this branch.
        va, vb = a.valid, b.valid
        value = va & vb ? convert(T, o.op(a.value, b.value)) : ifelse(va, a.value, b.value)
        return _Lane{T}(value, va | vb)
    end
    _valid(a) || return b
    _valid(b) || return a
    return _Lane{T}(o.op(a.value, b.value))
end

# The map `f` of a reduction pass whose inputs are already partial results.
struct _Partials end

# `op`, with its result converted to the accumulator type `A`: partial results keep that type
# (the accumulator rule makes this a no-op; a narrower `acctype` needs it)
struct _AccOp{A, OP}
    op::OP
end
_AccOp{A}(op::OP) where {A, OP} = _AccOp{A, OP}(op)
@inline (o::_AccOp{A})(a, b) where {A} = convert(A, o.op(a, b))

# The map and operator a reduction applies for the given seed: `f` and `op` (converting to the
# seed's type, the accumulator type) after a neutral element, lane-wrapped after an empty `_Lane`.
# Kernels call this on the device, so the wrappers never need to be converted for a backend.
@inline _lanefuncs(f, op, neutral::A) where {A} = (f, _AccOp{A}(op))
@inline _lanefuncs(::_Partials, op, neutral::A) where {A} = (identity, _AccOp{A}(op))
@inline _lanefuncs(f, op, ::_Lane{T}) where {T} = (_LaneMap{T}(f, op), _LaneOp(op))
@inline _lanefuncs(::_Partials, op, ::_Lane) = (identity, _LaneOp(op))

@inline _unlane(x) = x
@inline _unlane(x::_Lane) = x.value

# The seed of every partial result with accumulator type `A`: the caller's `neutral`, else
# GPUArraysCore's neutral element for `op` when it has one, else an empty lane. A seed that is not
# a lane stands for the accumulator type, so an abstract one (only on the host) is a lane: holding
# the caller's `neutral`, or empty.
function _reduce_seed(op, ::Type{A}, neutral) where {A}
    isconcretetype(A) || return neutral === nothing ? _Lane{A}() : _Lane{A}(neutral)
    neutral === nothing || return convert(A, neutral)
    Base.promote_op(neutral_element, Core.Typeof(op), Type{A}) === Union{} && return _Lane{A}()
    return _exact_neutral(op, convert(A, neutral_element(op, A)))
end

# GPUArraysCore's neutral element of `+` is `zero(T)`, which turns a floating-point sum of
# negative zeros into a positive zero; the identity of floating-point addition is `-0.0`
_exact_neutral(op, n) = n
_exact_neutral(::Union{typeof(+), typeof(Base.add_sum)},
               n::Union{AbstractFloat, Complex{<:AbstractFloat}}) = iszero(n) ? -n : n

# `init` of the kernels that write a destination: a value, `_NoInit()`, or `_Fold()` to fold in
# the destination's previous value as `Base.mapreducedim!` does.
struct _Fold end

# The value stored for an output whose reduction produced `partial`: `init` is applied exactly
# once, here.
@inline _finish(op, init, dst, i, partial) = op(init, _unlane(partial))
@inline _finish(op, ::_NoInit, dst, i, partial) = _unlane(partial)
@inline _finish(op, ::_Fold, dst, i, partial) = op(dst[i], _unlane(partial))


# Unrolled map constructing a tuple
@inline function unrolled_map_index(f, tuple_vector::Tuple)
    _unrolled_map_index(f, tuple_vector, (), 1)
end


@inline function _unrolled_map_index(f, rest::Tuple{}, acc, i)
    acc
end


@inline function _unrolled_map_index(f, rest::Tuple, acc, i)
    result = f(i)
    _unrolled_map_index(f, Base.tail(rest), (acc..., result), i + 1)
end


# Reductions whose every output reduces a single element, a partial result of the accumulator
# type `A`: `dst` and `src` have the same length.
function _mapreduce_nd_single!(f, op, dst, src, backend, ::Type{A}; init, launch...) where {A}
    _foreachindex(eachindex(dst), backend; launch...) do i
        x = Base.mapreduce_first(f, op, src[i])
        dst[i] = _finish(op, init, dst, i, A === Union{} ? x : convert(A, x))
    end
end

# Tree-reduce the `@groupsize()[1]` values in `sdata` into `sdata[1]`; the group size must be a power
# of two. All threads in the group must call it, after writing their value and synchronising.
@inline function reduce_group!(@context, op, sdata, ithread)
    # The group size is static, so this loop has a constant trip count; the conversion keeps the
    # index arithmetic in `ithread`'s integer type.
    s = typeof(ithread)(@groupsize()[1] >> 1)
    while s > 0x0
        if ithread < s
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + s + 0x1])
        end
        @synchronize()
        s >>= 0x1
    end
end
