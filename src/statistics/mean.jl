"""
    mean(
        f, src::AbstractArray{T}, backend::Backend=get_backend(src);
        dims::Union{Nothing, Int}=nothing,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    ) where {T<:Real}

    Compute the mean of `src` along dimensions `dims` after applying `f`.
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
function mean(
    f::Function,src::AbstractArray{T},backend::Backend=get_backend(src);
    dims::Union{Nothing, Int}=nothing,
    # CPU settings - ignored here
    max_tasks::Int = Threads.nthreads(),
    min_elems::Int = 1,
    prefer_threads::Bool=true,
    # GPU settings
    block_size::Int = 256,
    temp::Union{Nothing, AbstractArray} = nothing,
    switch_below::Int=0,
)  where {T<:Real}  
    init = T<:Integer ? zero(Float32) : zero(T)
    res = mapreduce(f,+,src,backend;
            init=init,
            dims=dims,
            max_tasks=max_tasks,
            min_elems=min_elems,
            prefer_threads=prefer_threads,
            block_size=block_size,
            temp=temp,
            switch_below=switch_below)
    if isnothing(dims)
        return res./length(src)
    else
        return res./size(src,dims)
    end
end

function mean(
    src::AbstractArray{T},backend::Backend=get_backend(src);
    dims::Union{Nothing, Int}=nothing,
    # CPU settings - ignored here
    max_tasks::Int = Threads.nthreads(),
    min_elems::Int = 1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int = 256,
    temp::Union{Nothing, AbstractArray} = nothing,
    switch_below::Int=0,
)  where {T<:Real}  
    init = T<:Integer ? zero(Float32) : zero(T)
    res = reduce(+,src,backend;
            init=init,
            dims=dims,
            max_tasks=max_tasks,
            min_elems=min_elems,
            prefer_threads=prefer_threads,
            block_size=block_size,
            temp=temp,
            switch_below=switch_below)
    if isnothing(dims)
        return res./length(src)
    else
        return res./size(src,dims)
    end
end
