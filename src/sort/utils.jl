# NOTE: for many index calculations in this library, computation using zero-indexing leads to
# fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
# indexing). Internal calculations will be done using zero indexing except when actually
# accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.
#
# If you use these functions, you'll have to offset the returned indices too.
Base.@propagate_inbounds function _lower_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(arr[right], value) && return right
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(arr[mid], value)
            left = mid
        else
            right = mid
        end
    end
    return left
end


Base.@propagate_inbounds function _upper_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(value, arr[left + 0x1]) && return left
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(value, arr[mid + 0x1])
            right = mid
        else
            left = mid
        end
    end
    return right
end
