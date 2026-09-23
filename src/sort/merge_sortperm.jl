# GPU merge sort permutation: sorts a copy of the keys, carrying the indices along.
function _merge_sortperm!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    inplace::Bool=false,
    block_size::Int=256,
    temp_ix::Union{Nothing, AbstractArray}=nothing,
    temp_v::Union{Nothing, AbstractArray}=nothing,
    dims::Union{Colon, Integer}=Colon(),
)
    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(ix) == length(v)
    dims isa Colon || @argcheck axes(ix) == axes(v)
    layout = slice_layout(v, dims)
    Base.Order.ord(lt, by, rev, order)      # validate the ordering keywords before touching ix
    if !isnothing(temp_ix)
        @argcheck length(temp_ix) == length(ix)
        @argcheck eltype(temp_ix) === eltype(ix)
    end

    if !isnothing(temp_v)
        @argcheck length(temp_v) == length(v)
        @argcheck eltype(temp_v) === eltype(v)
    end

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex(ix, backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(v) || layout.len <= 1) && return ix
    keys = inplace ? v : _copy(backend, v)

    _merge_sort_by_key!(
        keys, ix, backend;
        lt, by, rev, order, block_size, dims,
        temp_keys=temp_v, temp_values=temp_ix,
    )

    ix
end


# GPU merge sort permutation comparing the keys in global memory, without copying them.
function _merge_sortperm_lowmem!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    dims::Union{Colon, Integer}=Colon(),
)
    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(ix) == length(v)
    dims isa Colon || @argcheck axes(ix) == axes(v)
    layout = slice_layout(v, dims)
    ord = Base.Order.ord(lt, by, rev, order)
    if !isnothing(temp)
        @argcheck length(temp) == length(ix)
        @argcheck eltype(temp) === eltype(ix)
    end

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex(ix, backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(ix) || layout.len <= 1) && return ix

    # Construct custom comparator indexing into global array v
    comp = (ix, iy) -> Base.Order.lt(ord, v[ix], v[iy])

    # Block level
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_block!(backend, block_size)(
        ix, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level
    half_size_group = Int32(block_size * 2)
    size_group = half_size_group * 2
    if len > half_size_group
        p1 = ix
        p2 = isnothing(temp) ? similar(ix) : temp

        kernel! = _merge_sort_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(
                p1, p2, comp, half_size_group, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto!(ix, p1)
        end
    end

    ix
end
