function bucket_counts!(
    v::AbstractArray{T}, ord,
    splitters::Vector{T}, histograms::Matrix{Int},
    itask, irange,
) where T
    @inbounds begin
        for i in irange
            ibucket = 1 + searchsorted_last(splitters, v[i], 1, length(splitters), ord)
            histograms[ibucket, itask] += 1
        end
    end
    nothing
end


function bucket_offsets!(histograms, max_tasks)
    # Not worth parallelising this, as the number of tasks is much smaller than the number of
    # elements - in profiling this does not show up
    @inbounds begin

        # Sum up histograms and compute global offsets for each task
        offsets = @view histograms[1:max_tasks, max_tasks + 1]
        for itask in 1:max_tasks
            for j in 1:max_tasks
                offsets[j] += histograms[j, itask]
            end
        end
        accumulate!(+, offsets; init=0, inclusive=false,
                    alg=CPUThreads.Partitioned(max_tasks=1))

        # Compute each task's local offset into each bucket
        for itask in 1:max_tasks
            accumulate!(
                +, @view(histograms[itask, 1:max_tasks]),
                init=0,
                inclusive=false,
                alg=CPUThreads.Partitioned(max_tasks=1),
            )
        end
    end

    offsets
end


function scatter_buckets!(
    v, temp, ord,
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
            ibucket = 1 + searchsorted_last(splitters, v[i], 1, length(splitters), ord)

            # Get the current destination index for this element, then increment
            temp[offsets[ibucket]] = v[i]
            offsets[ibucket] += 1
        end
    end

    nothing
end


function sort_bucket!(v, temp, offsets, itask, max_tasks, ord)
    @inbounds begin
        istart = offsets[itask] + 1
        istop = itask == max_tasks ? length(temp) : offsets[itask + 1]

        if istart == istop
            v[istart] = temp[istart]
            return
        elseif istart > istop
            return
        end

        # At the end we will have to move elements from temp back to v anyways; for every
        # odd-numbered itask, move elements first, to avoid false sharing from threads
        if isodd(itask)
            copyto!(v, istart, temp, istart, istop - istart + 1)
            Base.sort!(view(v, istart:istop); order=ord)
        else
            # For even-numbered itasks, sort first, then move elements back to v
            Base.sort!(view(temp, istart:istop); order=ord)
            copyto!(v, istart, temp, istart, istop - istart + 1)
        end
    end

    return
end


function sample_sort_parallel!(v, temp, ord, splitters, histograms, max_tasks)
    # Compute the histogram for each task
    tp = TaskPartitioner(length(v), max_tasks, 1)
    itask_partition(tp) do itask, irange
        bucket_counts!(
            v, ord,
            splitters, histograms,
            itask, irange,
        )
    end

    # Compute the global and local (per-bucket) offsets for each task
    bucket_offsets!(histograms, max_tasks)
    offsets = @view histograms[1:max_tasks, max_tasks + 1]

    # Move the elements into the destination buffer
    itask_partition(tp) do itask, irange
        scatter_buckets!(
            v, temp, ord,
            splitters, offsets, histograms,
            itask, max_tasks, irange,
        )
    end

    # Sort each bucket in parallel
    itask_partition(tp) do itask, irange
        sort_bucket!(v, temp, offsets, itask, max_tasks, ord)
    end

    nothing
end




# Parallel sample sort on Julia threads, deferring to `Base.sort!` for the local sorts.
function sample_sort!(v::AbstractArray, ord::Base.Order.Ordering; max_tasks, min_elems, temp)
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
        # `temp` (a workspace's) is Base's scratch then
        return temp isa Vector{eltype(v)} && length(temp) >= num_elements ?
            Base.sort!(v; order=ord, scratch=temp) : Base.sort!(v; order=ord)
    end

    # Take equally spaced samples, save them in temp for the moment
    num_samples = oversampling_factor * max_tasks
    @inbounds for i in 1:num_samples
        temp[i] = v[1 + div((i - 1) * (num_elements - 1), num_samples - 1, RoundUp)]
    end

    # Sort samples and choose splitters; these are small allocations, which Julia is fast at
    Base.sort!(view(temp, 1:num_samples); order=ord)
    splitters = Vector{eltype(v)}(undef, max_tasks - 1)
    for i in 1:(max_tasks - 1)
        splitters[i] = temp[div(i * num_samples, max_tasks)]
    end

    # Pre-allocate histogram for each task; each column is exclusive to the task; one extra column
    # for global offsets; add 8 rows (i.e. 64 bytes) of padding to avoid false sharing
    histograms = zeros(Int, max_tasks + 8, max_tasks + 1)

    sample_sort_parallel!(v, temp, ord, splitters, histograms, max_tasks)

    v
end




# Sort permutation on Julia threads: sample sort of the indices, comparing the values.
function sample_sortperm!(ix::AbstractArray, v::AbstractArray, ord::Base.Order.Ordering;
                          max_tasks, min_elems, temp)
    # Initialise indices that will be sorted by the keys in v
    foreachindex_on(eachindex(ix), HOST_BACKEND; max_tasks, min_elems) do i
        @inbounds ix[i] = i
    end

    sample_sort_barrier!(ix, v, ord; max_tasks, min_elems, temp)
end


function sample_sort_barrier!(ix, v, ord; max_tasks, min_elems, temp)
    # Construct custom comparator indexing into global array v for every index comparison
    comp = (ix, iy) -> Base.Order.lt(ord, v[ix], v[iy])
    sample_sort!(ix, Base.Order.Lt(comp); max_tasks, min_elems, temp)
end


# Sort permutation of each slice along `dims`, like Base: sort the linear indices of each slice by
# the values they point to.
function sample_sortperm_dims!(ix, v, ord, dims; max_tasks, min_elems)
    perm_ord = Base.Order.Perm(ord, vec(v))
    copyto!(ix, LinearIndices(v))
    foreach_slice(ix, dims; max_tasks, min_elems) do slice
        Base.sort!(slice; order=perm_ord)
    end
    ix
end


# Key/value sort on Julia threads: find the stable sorting permutation of the keys (of each slice
# along `dims`), then gather keys and values through it.
function sample_sort_by_key!(keys, values, ord, dims, bufs; max_tasks, min_elems)
    ix = bufs.ix
    if dims isa Colon
        sample_sortperm!(vec(ix), vec(keys), ord; max_tasks, min_elems, temp=bufs.perm.temp)
    else
        sample_sortperm_dims!(ix, keys, ord, dims; max_tasks, min_elems)
    end
    gather!(keys, ix, bufs.keys; max_tasks, min_elems)
    gather!(values, ix, bufs.values; max_tasks, min_elems)
    keys, values
end

# v[i] = v_old[ix[i]] for every linear index i
function gather!(v, ix, old; max_tasks, min_elems)
    copyto!(old, v)
    task_partition(length(v), max_tasks, min_elems) do irange
        @inbounds for i in irange
            v[i] = old[ix[i]]
        end
    end
    v
end


# The scratch of `sample_sort!` over `n` elements of type `T`: the sorted output when the sort is
# parallel, `Base.sort!`'s scratch when it is not
sample_sort_scratch(a::CPUThreads.SampleSort, ::Type{T}, n) where {T} = (; temp=scratch(T, n))
