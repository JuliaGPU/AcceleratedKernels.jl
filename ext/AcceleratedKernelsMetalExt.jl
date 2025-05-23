module AcceleratedKernelsMetalExt


using Metal
import AcceleratedKernels as AK


# On Metal use the ScanPrefixes accumulation algorithm by default as the DecoupledLookback algorithm
# cannot be supported due to Metal's weaker memory consistency guarantees.
function AK.accumulate!(
    op, v::AbstractArray, backend::MetalBackend;
    init,
    neutral=AK.neutral_element(op, eltype(v)),
    dims::Union{Nothing, Int}=nothing,
    inclusive::Bool=true,

    # CPU settings - not used
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    AK._accumulate_impl!(
        op, v, backend,
        init=init, neutral=neutral, dims=dims, inclusive=inclusive,
        alg=alg,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


# Two-argument version for compatibility with Base.accumulate!
function AK.accumulate!(
    op, dst::AbstractArray, src::AbstractArray, backend::MetalBackend;
    init,
    neutral=AK.neutral_element(op, eltype(dst)),
    dims::Union{Nothing, Int}=nothing,
    inclusive::Bool=true,

    # CPU settings - not used
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    copyto!(dst, src)
    AK._accumulate_impl!(
        op, dst, backend,
        init=init, neutral=neutral, dims=dims, inclusive=inclusive,
        alg=alg,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


function AK.cumsum(
    src::AbstractArray, backend::MetalBackend;
    init=zero(eltype(src)),
    neutral=zero(eltype(src)),
    dims::Union{Nothing, Int}=nothing,

    # CPU settings - not used
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    AK.accumulate(
        +, src, backend;
        init=init,
        neutral=neutral,
        dims=dims,
        inclusive=true,

        alg=alg,

        block_size=block_size,
        temp=temp,
        temp_flags=temp_flags,
    )
end


function AK.cumprod(
    src::AbstractArray, backend::MetalBackend;
    init=one(eltype(src)),
    neutral=one(eltype(src)),
    dims::Union{Nothing, Int}=nothing,

    # CPU settings - not used
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    AK.accumulate(
        *, src, backend;
        init=init,
        neutral=neutral,
        dims=dims,
        inclusive=true,

        alg=alg,

        block_size=block_size,
        temp=temp,
        temp_flags=temp_flags,
    )
end


end   # module AcceleratedKernelsMetalExt