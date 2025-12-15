@inline function _chan_merge(a::Tuple{Int64,T,T}, b::Tuple{Int64,T,T}) where {T<:Real}
    nA, mA, M2A = a
    nB, mB, M2B = b
    if nA == 0
        return b
    elseif nB == 0
        return a
    else
        nAB  = nA + nB                          
        δ    = mB - mA                          
        invn = inv(T(nAB))                
        mean = (nA*mA + nB*mB) * invn
        cross = (δ*δ) * (nA*nB * invn)
        return (nAB, mean, M2A + M2B + cross)
    end
end
"""
    var(
        src::AbstractArray{T}, backend::Backend=get_backend(src);
        dims::Union{Nothing, Int}=nothing,
        corrected ::Bool = true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    ) where {T<:Real}

    Compute the varience of `src` along dimensions `dims`.
    If `dims` is `nothing`, reduce `src` to a scalar. If `dims` is an integer, reduce `src` along that
    dimension. The return type will be the same as the element type of `src` if it is a float type, or `Float32`
    if it is an integer type.
    ## CPU settings
    Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
    arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
    cases are scaling linearly with the number of threads.

    ## GPU settings
    The `block_size` parameter controls the number of threads per block.

    The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
    (`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
    required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
    array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
    except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
    zero, check against `Base.reduce` for CPU arrays for exact behavior.

    The `switch_below` parameter controls the threshold below which the reduction is performed on the
    CPU and is only used for 1D reductions (i.e. `dims=nothing`).
"""
function var(
    src::AbstractArray{T,N}, backend::Backend=get_backend(src);
    dims::Union{Nothing,Int}=nothing,
    corrected::Bool=true,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    block_size::Int=256,
    temp::Union{Nothing,AbstractArray}=nothing,  # ignored
    switch_below::Int=0,
) where {T<:Integer,N}

    init   = (0, 0f0, 0f0)
    mapper = x -> (1, Float32(x), 0f0)

    stats = mapreduce(
        mapper, _chan_merge, src, backend;
        init=init, neutral=init,
        dims=dims,
        max_tasks=max_tasks, min_elems=min_elems, prefer_threads=prefer_threads,
        block_size=block_size,
        temp=nothing,
        switch_below=switch_below,
    )

    if dims === nothing
        n, _, M2 = stats
        return M2 / Float32(n - ifelse(corrected , 1 , 0))
    else
        out = similar(stats, Float32)
        AcceleratedKernels.map!(
            s -> @inbounds(s[3] / Float32(s[1] - ifelse(corrected , 1 , 0))),
            out, stats, backend;
            max_tasks=max_tasks, min_elems=min_elems, block_size=block_size,
        )
        return out
    end
end


function var(
    src::AbstractArray{T,N}, backend::Backend=get_backend(src);
    dims::Union{Nothing,Int}=nothing,
    corrected::Bool=true,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    block_size::Int=256,
    temp::Union{Nothing,AbstractArray}=nothing,  # ignored
    switch_below::Int=0,
) where {T<:AbstractFloat,N}

    init   = (0, zero(T), zero(T))
    mapper = x -> (1, x, zero(typeof(x)))

    stats = mapreduce(
        mapper, _chan_merge, src, backend;
        init=init, neutral=init,
        dims=dims,
        max_tasks=max_tasks, min_elems=min_elems, prefer_threads=prefer_threads,
        block_size=block_size,
        temp=nothing,
        switch_below=switch_below,
    )

    if dims === nothing
        n, _, M2 = stats
        return M2 / T(n - ifelse(corrected , 1 , 0))
    else
        out = similar(stats, T)
        AcceleratedKernels.map!(
            s -> @inbounds(s[3] / (s[1] - ifelse(corrected , 1 , 0))),
            out, stats, backend;
            max_tasks=max_tasks, min_elems=min_elems, block_size=block_size,
        )
        return out
    end
end
