# GPU merge sort permutation: sorts a copy of the keys, `by.(v)`, carrying the indices along.
function merge_sortperm!(
    ix::AbstractArray, v::AbstractArray, backend::Backend, bufs::NamedTuple,
    ord::Base.Order.Ordering, by; block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex_on(Base.OneTo(length(ix)), backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(v) || layout.len <= 1) && return ix
    if by === identity
        # (a range, for instance, has no backend: collected on the host first)
        keys = copyto_on!(bufs.keys, backendless(v) ? collect(v) : v, backend)
        merge_sort_by_key!(keys, ix, backend, bufs, ord; block_size, dims)
    else
        # Evaluated once instead of in every comparison; `ord` is `By(by, ord.order)`
        keys = map!(by, bufs.keys, v; backend, block_size)
        merge_sort_by_key!(keys, ix, backend, bufs, ord.order; block_size, dims)
    end

    ix
end


# GPU merge sort permutation comparing the keys in global memory, without copying them.
function merge_sortperm_lowmem!(
    ix::AbstractArray, v::AbstractArray, backend::Backend, bufs::NamedTuple,
    ord::Base.Order.Ordering; block_size::Int, dims::Union{Colon, Integer},
)
    layout = slice_layout(v, dims)

    # Initialise the linear indices that will be sorted by the keys in v
    foreachindex_on(Base.OneTo(length(ix)), backend; block_size) do i
        @inbounds ix[i] = i
    end
    (isempty(ix) || layout.len <= 1) && return ix

    # Construct custom comparator indexing into global array v
    comp = (ix, iy) -> Base.Order.lt(ord, v[ix], v[iy])

    # Block level
    len = layout.len
    blocks = (len + block_size * 2 - 1) ÷ (block_size * 2)
    merge_tile_kernel!(backend, block_size)(
        ix, comp, layout, blocks,
        ndrange=(block_size * blocks * slice_count(layout),),
    )

    # Global level
    # (an `Int`: in slices of more than 2^30 elements, the run length doubles past `typemax(Int32)`)
    run_len = block_size * 2
    if len > run_len
        p1 = ix
        p2 = bufs.temp

        kernel! = merge_pass_kernel!(backend, block_size)

        niter = 0
        while len > run_len
            blocks = ((len + run_len - 1) ÷ run_len + 1) ÷ 2 * (run_len ÷ block_size)
            kernel!(
                p1, p2, comp, run_len, layout, blocks,
                ndrange=(block_size * blocks * slice_count(layout),),
            )

            run_len = run_len << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto_on!(ix, p1, backend)
        end
    end

    ix
end
