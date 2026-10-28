# The reduction contract's machinery (see `mapreducedim!`): the accumulator type, the value that
# seeds partial results, and how a finished partial becomes an output value.

# The mapped element type of a reduction source. Base's `combine_eltypes` does not see through
# the `Extruded` wrappers of a preprocessed `Broadcasted`, so its element types are derived here.
mapped_eltype(f::F, src) where {F} = Base.promote_op(f, source_eltype(src))
source_eltype(src::AbstractArray) = eltype(src)
source_eltype(bc::Base.Broadcast.Broadcasted) = Base.promote_op(bc.f, Base.map(source_eltype, bc.args)...)
source_eltype(x::Base.Broadcast.Extruded) = eltype(x.x)
source_eltype(x) = Base.Broadcast._broadcast_getindex_eltype(x)
# A `Broadcasted` source as a dense array on `backend` of its element type (`materialize` would
# take the element type from `combine_eltypes`, which gives `Any` for `Extruded` arguments). A
# kernel evaluates it, so that its arrays need not have a broadcast style of the backend.
function materialize_source(bc::Base.Broadcast.Broadcasted, backend)
    dst = KernelAbstractions.allocate(backend, source_eltype(bc), size(bc))
    indices = CartesianIndices(axes(bc))
    foreachindex_on(Base.OneTo(length(dst)), backend) do i
        @inbounds dst[i] = bc[indices[i]]
    end
    return dst
end

# The type of a reduction's first partial value when no `init` is given.
first_type(op::OP, ::Type{M}) where {OP, M} =
    Base.promote_op(Base.reduce_first, Core.Typeof(op), M)

# The accumulator type of a reduction of mapped elements of type `M` into `S` (the type of `init`
# or of the destination, `Union{}` for neither): the types Base's fold `op(op(S, M), M)...` goes
# through, joined with `promote_type`. In a parallel reduction every element is also a
# one-element partial result (`Base.reduce_first(op, x)`), and partial results are combined with
# each other, so those types are joined as well; `S` itself is not a partial result (`init` and
# the destination's values are applied once, at the end). So `sum(Int8[...])` accumulates in
# `Int`, as Base's `add_sum` does, and partial results have one type however many elements they
# combine. `Union{}` when inference shows that the fold always throws; only a reduction with
# elements to combine is an error then (`check_combinable`).
function accumulator_type(op::OP, ::Type{S}, ::Type{M}) where {OP, S, M}
    A = first_type(op, M)
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

check_combinable(op, f, ::Type{A}) where {A} = A === Union{} && throw(ArgumentError(
    "`$op` cannot combine the mapped elements (`$f`): calling it always throws for their types"))

# The accumulator type of a reduction: the caller's `acctype`, else the rule of
# `accumulator_type`. An `acctype` is rejected only where the types rule it out, when a value that
# a partial result takes (`Base.reduce_first`'s of an element, or `op`'s of a partial result and
# an element or of two partial results) has no conversion to it at all; whether the values fit is
# the caller's obligation.
resolve_acctype(op::OP, ::Type{S}, ::Type{M}, ::Nothing) where {OP, S, M} = accumulator_type(op, S, M)
function resolve_acctype(op::OP, ::Type{S}, ::Type{M}, ::Type{A}) where {OP, S, M, A}
    M === Union{} && return A           # (no elements to hold)
    check_holds(A, first_type(op, M), "a one-element partial result (`Base.reduce_first`)")
    for (T, what) in ((Base.promote_op(op, A, M), "a partial result and an element"),
                      (Base.promote_op(op, A, A), "two partial results"))
        T === Union{} && throw(ArgumentError("`acctype=$A`: `$op` of $what always throws"))
        check_holds(A, T, "`$op` of $what")
    end
    return A
end
check_holds(::Type{A}, ::Type{T}, what) where {A, T} =
    T === Union{} || Base.promote_op(convert, Type{A}, T) !== Union{} ||
    throw(ArgumentError("`acctype=$A` cannot hold $what, of type `$T`"))
resolve_acctype(op, ::Type, ::Type, acctype) =
    throw(ArgumentError("`acctype` must be a type or `nothing`, got $(repr(acctype))"))

# A partial result of a reduction whose operator has no known neutral element. It starts out
# unset and takes its first value from `Base.mapreduce_first`; combining with an unset one
# returns the other operand, keeping the order of the operands. For associative operators this
# gives the same result as seeding with a neutral element.
#
# An unset `Optional` of a bits type (every one kernels see) holds zero bits; one of another type
# leaves the value undefined, which is how it is recognised. The flag `F` is `flag_type(T)`.
struct Optional{T, F}
    value::T
    isset::F
    Optional{T, F}() where {T, F} = new{T, F}()
    Optional{T, F}(value, isset::Bool) where {T, F} = new{T, F}(value, isset)
end
Optional{T}() where {T} =
    isinline(T) ? Optional{T, flag_type(T)}(zero_bits(T), false) : Optional{T, flag_type(T)}()
Optional{T}(value) where {T} = Optional{T, flag_type(T)}(value, true)
Optional{T}(value, isset::Bool) where {T} = Optional{T, flag_type(T)}(value, isset)

# The flag of an `Optional{T}`. A `Bool` would leave the bytes up to `T`'s alignment as padding,
# which stores of an `Optional` don't write but kernels load (partials are loaded whole) and the
# host copies, so Compute Sanitizer's initcheck reports reading them. For a bits type, the flag is
# the unsigned integer that fills those bytes, up to 8 of them: OpenCL's SPIR-V has no wider
# integers, so an `Optional` of a 16-byte aligned type keeps some padding (as does a `T` with
# padding of its own).
@generated function flag_type(::Type{T}) where {T}
    isbitstype(T) || return Bool
    return (UInt8, UInt16, UInt32, UInt64)[trailing_zeros(min(Base.datatype_alignment(T), 8)) + 1]
end

isinline(::Type{T}) where {T} = isbitstype(T) || Base.isbitsunion(T)
@inline isset(l::Optional{T}) where {T} = isinline(T) ? !iszero(l.isset) : isdefined(l, :value)

# A value of the bits type or bits union `T` with all-zero bits
function zero_bits(::Type{T}) where {T}
    isbitstype(T) || return zero_bits(first(Base.uniontypes(T)))
    Base.issingletontype(T) && return T.instance
    bytes = zeros(UInt8, sizeof(T))
    return GC.@preserve bytes unsafe_load(Ptr{T}(pointer(bytes)))
end

struct OptionalMap{F, OP, T}
    f::F
    op::OP
end
OptionalMap{T}(f::F, op::OP) where {T, F, OP} = OptionalMap{F, OP, T}(f, op)
@inline (m::OptionalMap{F, OP, T})(x) where {F, OP, T} = Optional{T}(Base.mapreduce_first(m.f, m.op, x))

struct OptionalOp{OP}
    op::OP
end
@inline function (o::OptionalOp)(a::Optional{T}, b::Optional{T}) where {T}
    if !isprimitivetype(T)
        isset(a) || return b
        isset(b) || return a
        return Optional{T}(o.op(a.value, b.value))
    end
    # One result built from the operands' fields: returning either operand made the compiler
    # merge the possible results through stack slots. (Only for primitive types: Julia selects
    # between other values, e.g. tuples, through stack slots too, once per combine.)
    sa, sb = isset(a), isset(b)
    value = sa & sb ? convert(T, o.op(a.value, b.value)) : ifelse(sa, a.value, b.value)
    return Optional{T}(value, sa | sb)
end

# The map `f` of a reduction pass whose inputs are already partial results.
struct PartialInputs end

# `op`, with its result converted to the accumulator type `A`: partial results keep that type
# (the accumulator rule makes this a no-op; a narrower `acctype` needs it)
struct AccOp{A, OP}
    op::OP
end
AccOp{A}(op::OP) where {A, OP} = AccOp{A, OP}(op)
@inline (o::AccOp{A})(a, b) where {A} = convert(A, o.op(a, b))

# The map and operator a reduction applies for the given seed: `f` and `op` (converting to the
# seed's type, the accumulator type) after a neutral element, wrapped in `Optional`s after an unset
# one.
# Kernels call this on the device, so the wrappers never need to be converted for a backend.
@inline seeded_ops(f, op, seed::A) where {A} = (f, AccOp{A}(op))
@inline seeded_ops(::PartialInputs, op, seed::A) where {A} = (identity, AccOp{A}(op))
@inline seeded_ops(f, op, ::Optional{T}) where {T} = (OptionalMap{T}(f, op), OptionalOp(op))
@inline seeded_ops(::PartialInputs, op, ::Optional) = (identity, OptionalOp(op))

@inline unwrap(x) = x
@inline unwrap(x::Optional) = x.value

# Scans keep elements and partial results in one array or tile: an element enters the seed's
# representation with `lift` (partials pass through), and a partial is stored in an array of
# element type `E` with `lower` (as an `Optional` only in arrays of them).
@inline lift(seed, x) = x
@inline lift(::Optional{T}, x) where {T} = Optional{T}(x)
@inline lift(::Optional{T}, x::Optional{T}) where {T} = x
@inline lower(::Type{E}, p) where {E} = unwrap(p)
@inline lower(::Type{E}, p) where {E <: Optional} = p

# Store the partial `p` in `v[i]`, except an unset one where elements are stored (a scan's
# carry pass fills that element later; its value field may not even be defined)
@inline function store_partial!(v, i, p)
    if !(p isa Optional) || eltype(v) <: Optional || isset(p)
        @inbounds v[i] = lower(eltype(v), p)
    end
    return nothing
end

# The seed of every partial result with accumulator type `A`: the caller's `neutral`, else
# GPUArraysCore's neutral element for `op` when it has one, else an unset `Optional`. A seed that
# is not an `Optional` stands for the accumulator type, so an abstract one (only on the host) is an
# `Optional`: holding the caller's `neutral`, or unset.
function reduce_seed(op::OP, ::Type{A}, neutral) where {OP, A}
    isconcretetype(A) || return neutral === nothing ? Optional{A}() : Optional{A}(neutral)
    neutral === nothing || return convert(A, neutral)
    Base.promote_op(neutral_element, Core.Typeof(op), Type{A}) === Union{} && return Optional{A}()
    return exact_neutral(op, convert(A, neutral_element(op, A)))
end

# GPUArraysCore's neutral element of `+` is `zero(T)`, which turns a floating-point sum of
# negative zeros into a positive zero; the identity of floating-point addition is `-0.0`
exact_neutral(op, n) = n
exact_neutral(::Union{typeof(+), typeof(Base.add_sum)},
              n::Union{AbstractFloat, Complex{<:AbstractFloat}}) = iszero(n) ? -n : n

# `init` of the kernels that write a destination: a value, `NoInit()`, or `FoldInto()` to fold in
# the destination's previous value as `Base.mapreducedim!` does.
struct FoldInto end

# The value stored for an output whose reduction produced `partial`: `init` is applied exactly
# once, here.
@inline finish(op, init, dst, i, partial) = op(init, unwrap(partial))
@inline finish(op, ::NoInit, dst, i, partial) = unwrap(partial)
@inline finish(op, ::FoldInto, dst, i, partial) = op(dst[i], unwrap(partial))

# `init` of a pass that stores bare partial results, for a later pass or for the host
struct Unfinished end
@inline finish(op, ::Unfinished, dst, i, partial) = partial
