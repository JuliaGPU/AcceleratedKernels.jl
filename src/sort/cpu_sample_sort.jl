function _sample_sort_histogram!(
    v::AbstractArray{T}, ord,
    splitters::Vector{T}, histograms::Matrix{Int},
    itask, irange,
) where T
    @inbounds begin
        for i in irange
            ibucket = 1 + _searchsortedlast(splitters, v[i], 1, length(splitters), ord)
            histograms[ibucket, itask] += 1
        end
    end
    nothing
end


function _sample_sort_compute_offsets!(histograms, max_tasks)
    @inbounds begin
        # Sum up histograms and compute global offsets for each task
        offsets = @view histograms[1:max_tasks, max_tasks + 1]
        for itask in 1:max_tasks
            for j in 1:max_tasks
                offsets[j] += histograms[j, itask]
            end
        end
        accumulate!(+, offsets, init=0, inclusive=false, max_tasks=1)

        # Compute each task's local offset into each bucket
        for itask in 1:max_tasks
            accumulate!(
                +, @view(histograms[itask, 1:max_tasks]),
                init=0,
                inclusive=false,
                max_tasks=1,
            )
        end
    end

    offsets
end


function _sample_sort_move_buckets!(
    v, dest, ord,
    splitters, global_offsets, task_offsets,
    itask, max_tasks, irange,
)
    # Copy the elements into the destination buffer following splitters
    @inbounds begin

        # Compute the destination indices for this task into each bucket
        offsets = @view task_offsets[1:max_tasks, itask]
        for it in 1:max_tasks
            offsets[it] += global_offsets[it] + 1
        end

        for i in irange
            # Find the bucket for this element
            ibucket = 1 + _searchsortedlast(splitters, v[i], 1, length(splitters), ord)

            # Get the current destination index for this element, then increment
            dest[offsets[ibucket]] = v[i]
            offsets[ibucket] += 1
        end
    end

    nothing
end


function _sample_sort_sort_bucket!(
    v, dest, offsets, itask, max_tasks;
    lt, by, rev, order    
)
    @inbounds begin
        istart = offsets[itask] + 1
        istop = itask == max_tasks ? length(dest) : offsets[itask + 1]

        if istart == istop
            v[istart] = dest[istart]
            return
        elseif istart > istop
            return
        end

        # At the end we will have to move elements from dest back to v anyways; for every
        # odd-numbered itask, move elements first, to avoid false sharing from threads
        if isodd(itask)
            copyto!(v, istart, dest, istart, istop - istart + 1)
            Base.sort!(view(v, istart:istop), lt=lt, by=by, rev=rev, order=order)
        else
            # For even-numbered itasks, sort first, then move elements back to v
            Base.sort!(view(dest, istart:istop), lt=lt, by=by, rev=rev, order=order)
            copyto!(v, istart, dest, istart, istop - istart + 1)
        end
    end

    return
end


function _sample_sort_parallel!(
    v, dest, ord,
    splitters, histograms,
    max_tasks;
    lt, by, rev, order,
)
    # Compute the histogram for each task
    tp = TaskPartitioner(length(v), max_tasks, 1)
    itask_partition(tp) do itask, irange
        _sample_sort_histogram!(
            v, ord,
            splitters, histograms,
            itask, irange,
        )
    end

    # Compute the global and local (per-bucket) offsets for each task
    _sample_sort_compute_offsets!(histograms, max_tasks)
    offsets = @view histograms[1:max_tasks, max_tasks + 1]

    # Move the elements into the destination buffer
    itask_partition(tp) do itask, irange
        _sample_sort_move_buckets!(
            v, dest, ord,
            splitters, offsets, histograms,
            itask, max_tasks, irange,
        )
    end

    # Sort each bucket in parallel
    itask_partition(tp) do itask, irange
        _sample_sort_sort_bucket!(
            v, dest, offsets, itask, max_tasks;
            lt=lt, by=by, rev=rev, order=order,
        )
    end

    # # Debug: single-threaded version
    # tp = TaskPartitioner(length(v), max_tasks, 1)
    # for itask in 1:max_tasks
    #     irange = tp[itask]
    #     _sample_sort_histogram!(
    #         v, ord,
    #         splitters, histograms,
    #         itask, irange,
    #     )
    # end
    # _sample_sort_compute_offsets!(histograms, max_tasks)
    # offsets = @view histograms[1:max_tasks, max_tasks + 1]
    # for itask in 1:max_tasks
    #     irange = tp[itask]
    #     _sample_sort_move_buckets!(
    #         v, dest, ord,
    #         splitters, offsets, histograms,
    #         itask, max_tasks, irange,
    #     )
    # end
    # for itask in 1:max_tasks
    #     _sample_sort_sort_bucket!(
    #         v, dest, offsets, itask, max_tasks;
    #         lt=lt, by=by, rev=rev, order=order,
    #     )
    # end

    nothing
end




"""
    sample_sort!(
        v::AbstractArray;

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        max_tasks=Threads.nthreads(),
        min_elems=1,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sample_sort!(
    v::AbstractArray;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    max_tasks=Threads.nthreads(),
    min_elems=1,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    # Sanity checks
    @argcheck max_tasks > 0

    # For uniform distributions, the error is O(1/sqrt(n)); still, there may be pathological
    # cases - maybe there's a fancier way to choose samples / splitters?
    oversampling_factor = 16
    num_elements = length(v)

    # Trivial cases
    if num_elements < 2
        return v
    end
    max_tasks = min(max_tasks, num_elements ÷ min_elems)
    if max_tasks <= 1 || num_elements < oversampling_factor * max_tasks
        return Base.sort!(v, lt=lt, by=by, rev=rev, order=order)
    end

    # Create a temporary buffer for the sorted output
    if isnothing(temp)
        dest = similar(v)
    else
        @argcheck length(temp) == length(v)
        @argcheck eltype(temp) == eltype(v)
        dest = temp
    end

    # Take equally spaced samples, save them in dest for the moment
    num_samples = oversampling_factor * max_tasks
    isamples = IntLinSpace(1, num_elements, num_samples)
    @inbounds for i in 1:num_samples
        dest[i] = v[isamples[i]]
    end

    # Sort samples and choose splitters; these are small allocations, which Julia is fast at
    Base.sort!(view(dest, 1:num_samples), lt=lt, by=by, rev=rev, order=order)
    splitters = Vector{eltype(v)}(undef, max_tasks - 1)
    for i in 1:(max_tasks - 1)
        splitters[i] = dest[div(i * num_samples, max_tasks)]
    end

    # Pre-allocate histogram for each task; each column is exclusive to the task; one extra column
    # for global offsets; add 8 rows (i.e. 64 bytes) of padding to avoid false sharing
    histograms = zeros(Int, max_tasks + 8, max_tasks + 1)

    # Run parallel sample sort for a given constructed ord (there may be a type instability for
    # rev=true, so we keep this function barrier)
    ord = Base.Order.ord(lt, by, rev, order)
    _sample_sort_parallel!(
        v, dest, ord,
        splitters, histograms,
        max_tasks;
        lt=lt, by=by, rev=rev, order=order,
    )

    v
end




"""
    sample_sortperm!(
        ix::AbstractArray, v::AbstractArray;

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        max_tasks=Threads.nthreads(),
        min_elems=1,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sample_sortperm!(
    ix::AbstractArray, v::AbstractArray;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    max_tasks=Threads.nthreads(),
    min_elems=1,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    # Sanity checks
    @argcheck max_tasks > 0
    @argcheck length(ix) == length(v)

    # Initialise indices that will be sorted by the keys in v
    foreachindex(ix, max_tasks=max_tasks) do i
        @inbounds ix[i] = i
    end

    # Construct custom comparator indexing into global array v for every index comparison
    ilt = (ix, iy) -> lt(v[ix], v[iy])

    # Sort with custom comparator
    sample_sort!(
        ix;
        lt=ilt, by=by, rev=rev, order=order,
        max_tasks=max_tasks,
        min_elems=min_elems,
        temp=temp,
    )
end
