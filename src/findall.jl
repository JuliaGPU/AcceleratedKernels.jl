# findall via stream compaction: inclusive-scan the predicate mask to get each selected element's
# output slot and the total count, then scatter. `keytype`/`CartesianIndices` yield Base-matching
# keys - `Int` for a 1-D input, `CartesianIndex{N}` for an N-D one.


"""
    findall(pred, v::AbstractArray, backend::Backend=get_backend(v); ...)
    findall(bools::AbstractArray{Bool}, backend::Backend=get_backend(bools); ...)

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size=256,

Return the indices of the `true` elements of `bools`, or of the elements of `v` satisfying `pred`,
in ascending order - a `Vector{Int}` for a 1-D input and a `Vector{CartesianIndex{N}}` for an N-D
one, matching `Base.findall`. The CPU and GPU settings are the same as for [`foreachindex`](@ref).

# Examples
```julia
import CUDA
import AcceleratedKernels as AK

v = CUDA.CuArray(Int32[5, -2, 8, -1, 3])
AK.findall(x -> x > 0, v)               # [1, 3, 5]

m = CUDA.CuArray(Bool[1 0; 0 1])
AK.findall(m)                           # [CartesianIndex(1, 1), CartesianIndex(2, 2)]
```
"""
function findall(
    bools::AbstractArray{Bool}, backend::Backend=get_backend(bools);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # GPU settings
    block_size::Int=256,
    prefer_threads::Bool=true,
)
    I = keytype(bools)
    isempty(bools) && return similar(bools, I, 0)

    # indices[i] = output slot of element i when selected; indices[end] = number selected
    indices = accumulate(
        +, reshape(bools, length(bools)), backend;
        init=0, inclusive=true,
        max_tasks, min_elems=max(min_elems, 2), block_size, prefer_threads,
    )
    n = @allowscalar(indices[end])

    # slots are unique and increasing at selected positions, so the scatter is race-free
    ys = similar(bools, I, n)
    n > 0 && _findall_scatter!(
        ys, bools, indices, CartesianIndices(bools), backend;
        max_tasks, min_elems, block_size, prefer_threads,
    )
    ys
end


# Separate function so the `foreachindex` closure is type-stable.
function _findall_scatter!(ys, bools, indices, cartesian, backend; kwargs...)
    foreachindex(bools, backend; kwargs...) do i
        @inbounds if bools[i]
            ys[indices[i]] = cartesian[i]       # CartesianIndex{1} converts to Int for a vector
        end
    end
    ys
end


function findall(
    pred, v::AbstractArray, backend::Backend=get_backend(v);

    # CPU settings
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,

    # GPU settings
    block_size::Int=256,
    prefer_threads::Bool=true,
)
    # evaluate the predicate once into a Bool mask, then compact
    flags = similar(v, Bool)
    map!(pred, flags, v, backend; max_tasks, min_elems, block_size, prefer_threads)
    findall(flags, backend; max_tasks, min_elems, block_size, prefer_threads)
end
