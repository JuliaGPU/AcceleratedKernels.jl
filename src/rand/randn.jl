const ALLOWED_RANDN_SCALARS = Union{Float16, Float32, Float64}

const U24_MAX_SAFE_MIDPOINT = UInt32(0x00fffffe)                 # 2^24 - 2
const U53_MAX_SAFE_MIDPOINT = UInt64(0x001ffffffffffffe)         # 2^53 - 2
const MIDPOINT_SCALE_F32 = ldexp(Float32(1), -24)                # 2^-24
const MIDPOINT_SCALE_F64 = ldexp(Float64(1), -53)                # 2^-53




#=
The below Float constructions are not duplicates of those in utilities.jl - they are needed to
ensure an interval of (0, 1) as opposed to [0, 1). Achieving this purely logically with midpoint
mapping means we can avoid a check for producing a 0 (which would normally cause a redraw).
Avoiding 0 is essential for Box-Muller due to the logarithm functions.
=#


# Convert random UInt32 bits to Float32 in (0, 1) using midpoint mapping on a 24-bit grid.
@inline function uint32_to_open_unit_float32_midpoint(u::UInt32)::Float32
    # `min` keeps the top midpoint below one after Float32 rounding.
    k = min(u >> 8, U24_MAX_SAFE_MIDPOINT)
    return (Float32(k) + 0.5f0) * MIDPOINT_SCALE_F32
end


# Convert random UInt64 bits to Float64 in (0, 1) using midpoint mapping on a 53-bit grid.
@inline function uint64_to_open_unit_float64_midpoint(u::UInt64)::Float64
    # `min` keeps the top midpoint below one after Float64 rounding.
    k = min(u >> 11, U53_MAX_SAFE_MIDPOINT)
    return (Float64(k) + 0.5) * MIDPOINT_SCALE_F64
end


# Float16 path reuses Float32 midpoint sampling for robust math in Box-Muller.
@inline function rand_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float16},
)::Float16
    return Float16(rand_open01(seed, alg, counter, Float32))
end


@inline function rand_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float32},
)::Float32
    return uint32_to_open_unit_float32_midpoint(rand_uint(seed, alg, counter, UInt32))
end


@inline function rand_open01(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    counter::UInt64,
    ::Type{Float64},
)::Float64
    return uint64_to_open_unit_float64_midpoint(rand_uint(seed, alg, counter, UInt64))
end


@inline function rand_open01(::UInt64, ::CounterRNGAlgorithm, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported open-interval random type $(T). Supported: $(ALLOWED_RANDN_SCALARS)"
    ))
end


@inline function randn_pair(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    pair_counter::UInt64,
    ::Type{Float16},
)::Tuple{Float16, Float16}
    z0, z1 = randn_pair(seed, alg, pair_counter, Float32)
    return Float16(z0), Float16(z1)
end


@inline function randn_pair(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    pair_counter::UInt64,
    ::Type{Float32},
)::Tuple{Float32, Float32}
    u = rand_uint(seed, alg, pair_counter, UInt64)
    u1 = uint32_to_open_unit_float32_midpoint(_u32_lo(u))
    u2 = uint32_to_open_unit_float32_midpoint(_u32_hi(u))
    radius = sqrt(-2.0f0 * log(u1))
    theta = Float32(2pi) * u2
    stheta, ctheta = sincos(theta)
    return radius * ctheta, radius * stheta
end


@inline function randn_pair(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    pair_counter::UInt64,
    ::Type{Float64},
)::Tuple{Float64, Float64}
    c0 = pair_counter << 1
    u1 = rand_open01(seed, alg, c0, Float64)
    u2 = rand_open01(seed, alg, c0 + UInt64(1), Float64)
    radius = sqrt(-2.0 * log(u1))
    theta = Float64(2pi) * u2
    stheta, ctheta = sincos(theta)
    return radius * ctheta, radius * stheta
end


@inline function randn_pair(::UInt64, ::CounterRNGAlgorithm, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported normal random type $(T). Supported: $(ALLOWED_RANDN_SCALARS)"
    ))
end


@inline function randn_scalar(
    seed::UInt64,
    alg::CounterRNGAlgorithm,
    normal_counter::UInt64,
    ::Type{T},
)::T where {T <: ALLOWED_RANDN_SCALARS}
    pair_counter = normal_counter >> 1
    z0, z1 = randn_pair(seed, alg, pair_counter, T)
    return iszero(normal_counter & UInt64(0x1)) ? z0 : z1
end


@inline function randn_scalar(::UInt64, ::CounterRNGAlgorithm, ::UInt64, ::Type{T}) where {T}
    throw(ArgumentError(
        "Unsupported normal random scalar type $(T). Supported: $(ALLOWED_RANDN_SCALARS)"
    ))
end


"""
    randn!(
        rng::CounterRNG,
        v::AbstractArray{T},
        backend::Backend=get_backend(v);

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,

        # Implementation choice
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
    ) where {T <: AbstractFloat}

Fill `v` in-place with pseudo-random samples from a standard normal distribution.

For `v[i]`, the normal stream counter is `rng.offset + UInt64(i - 1)` in linear indexing order.
Values are generated using Box-Muller from midpoint-open uniforms in `(0, 1)`.

After filling `v`, `rng.offset` advances by `length(v)`.
"""
function randn!(
    rng::CounterRNG,
    v::AbstractArray{T},
    backend::Backend=get_backend(v);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int=256,
) where T

    @argcheck T <: ALLOWED_RANDN_SCALARS "Unsupported eltype $T. Supported: $(ALLOWED_RANDN_SCALARS)"

    isempty(v) && return v

    # Local isbits captures from mutable rng object.
    seed, alg, initial_offset = rng.seed, rng.alg, rng.offset
    len = length(v)
    pair_start = initial_offset >> 1

    # Even stream offset is the common path and maps pair `i` to output indices `(2i-1, 2i)`.
    if iszero(initial_offset & UInt64(0x1))
        pair_count = cld(len, 2)
        pair_indices = Base.OneTo(pair_count)

        # Fully branch-free hot path when both offset and length are even.
        if iseven(len)
            foreachindex(
                pair_indices, backend;
                max_tasks, min_elems, prefer_threads, block_size,
            ) do i
                pair_counter = pair_start + _counter_from_index(i)
                z0, z1 = randn_pair(seed, alg, pair_counter, T)
                i0 = (i << 1) - 1
                @inbounds v[i0] = z0
                @inbounds v[i0 + 1] = z1
            end
        else
            foreachindex(
                pair_indices, backend;
                max_tasks, min_elems, prefer_threads, block_size,
            ) do i
                pair_counter = pair_start + _counter_from_index(i)
                z0, z1 = randn_pair(seed, alg, pair_counter, T)
                i0 = (i << 1) - 1
                @inbounds v[i0] = z0
                i1 = i0 + 1

                if i1 <= len
                    @inbounds v[i1] = z1
                end
            end
        end
    else
        # Odd stream offset shifts pair `i` to `(2i-2, 2i-1)`; only the first z0 is out of range.
        pair_count = cld(len + 1, 2)
        pair_indices = Base.OneTo(pair_count)

        foreachindex(
            pair_indices, backend;
            max_tasks, min_elems, prefer_threads, block_size,
        ) do i
            pair_counter = pair_start + _counter_from_index(i)
            z0, z1 = randn_pair(seed, alg, pair_counter, T)
            i0 = (i << 1) - 2

            if i0 >= 1
                @inbounds v[i0] = z0
            end

            i1 = i0 + 1
            if i1 <= len
                @inbounds v[i1] = z1
            end
        end
    end

    rng.offset += UInt64(len)

    v
end


function randn!(
    v::AbstractArray,
    args...;
    kwargs...,
)
    return randn!(CounterRNG(), v, args...; kwargs...)
end
