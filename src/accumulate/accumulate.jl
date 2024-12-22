include("accumulate_1d.jl")


"""
    accumulate!(
        op, v::AbstractArray, backend::Backend=get_backend(v);
        init,
        inclusive::Bool=true,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Compute accumulated running totals along a sequence by applying a binary operator to all elements
up to the current one; often used in GPU programming as a first step in finding / extracting
subsets of data.

**Other names**: prefix sum, `thrust::scan`, cumulative sum; inclusive (or exclusive) if the first
element is included in the accumulation (or not).

The `block_size` should be a power of 2 and greater than 0. The temporaries `temp` and `temp_flags`
should both have at least
`(length(v) + 2 * block_size - 1) ÷ (2 * block_size)` elements; `eltype(v) === eltype(temp)`; the
elements in `temp_flags` can be any integers, but `Int8` is used by default to reduce memory usage. 

# Platform-Specific Notes
Currently, Apple Metal GPUs do not have strong enough memory consistency guarantees to support the
industry-standard "decoupled lookback" algorithm for prefix sums - which means it currently may,
for very large arrays, produce incorrect results ~0.38% of the time. We are currently working on an
alternative algorithm without lookback ([issue](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/10)).

The CPU implementation currently defers to the single-threaded Base.accumulate!; we are waiting on a
multithreaded implementation in OhMyThreads.jl ([issue](https://github.com/JuliaFolds2/OhMyThreads.jl/issues/129)).

# Examples
Example computing an inclusive prefix sum (the typical GPU "scan"):
```julia
import AcceleratedKernels as AK
using oneAPI

v = oneAPI.ones(Int32, 100_000)
AK.accumulate!(+, v, init=0)
```
"""
function accumulate!(
    op, v::AbstractArray, backend::Backend=get_backend(v);
    init,
    inclusive::Bool=true,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    _accumulate_impl!(
        op, v, backend,
        init=init, inclusive=inclusive,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


function _accumulate_impl!(
    op, v::AbstractArray, backend::Backend;
    init,
    inclusive::Bool=true,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    if backend isa GPU
        accumulate_1d!(
            op, v, backend,
            init=init, inclusive=inclusive,
            block_size=block_size, temp=temp, temp_flags=temp_flags,
        )
        return v
    else
        # Simple single-threaded CPU implementation - FIXME: implement taccumulate in OhMyThreads.jl
        if length(v) == 0
            return v
        end
        if inclusive
            running = v[begin]
            for i in firstindex(v) + 1:lastindex(v)
                running = op(running, v[i])
                v[i] = running
            end
        else
            running = init
            for i in eachindex(v)
                v[i], running = running, op(running, v[i])
            end
        end
        return v
    end
end


"""
    accumulate(
        op, v::AbstractArray, backend::Backend=get_backend(v);
        init,
        inclusive::Bool=true,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        temp_flags::Union{Nothing, AbstractArray}=nothing,
    )

Out-of-place version of [`accumulate!`](@ref).
"""
function accumulate(
    op, v::AbstractArray, backend::Backend=get_backend(v);
    init,
    inclusive::Bool=true,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    vcopy = copy(v)
    accumulate!(op, vcopy, backend; init=init, inclusive=inclusive,
                block_size=block_size, temp=temp, temp_flags=temp_flags)
    vcopy
end

