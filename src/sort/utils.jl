# NOTE: for many index calculations in this library, computation using zero-indexing leads to
# fewer operations (also code is transpiled to CUDA / ROCm / oneAPI / Metal code which do zero
# indexing). Internal calculations will be done using zero indexing except when actually
# accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.
#
# If you use these functions, you'll have to offset the returned indices too.
@inbounds @inline function _lower_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(@inbounds(arr[right]), value) && return right
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(@inbounds(arr[mid]), value)
            left = mid
        else
            right = mid
        end
    end
    return left
end


@inbounds @inline function _upper_bound_s0(arr, value, left=0, right=length(arr), comp=(<))
    right <= left && return left
    comp(value, @inbounds(arr[left + 0x1])) && return left
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(value, @inbounds(arr[mid + 0x1]))
            right = mid
        else
            left = mid
        end
    end
    return right
end


@inbounds @inline function _lower_bound_si0(ix, vec, value, left=0, right=length(ix), comp=(<))
    right <= left && return left
    comp(vec[@inbounds(ix[right])], value) && return right
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(vec[@inbounds(ix[mid])], value)
            left = mid
        else
            right = mid
        end
    end
    return left
end


@inbounds @inline function _upper_bound_si0(ix, vec, value, left=0, right=length(ix), comp=(<))
    right <= left && return left
    comp(value, vec[@inbounds(ix[left + 0x1])]) && return left
    while right > left + 0x1
        mid = left + ((right - left) >> 0x1)
        if comp(value, vec[@inbounds(ix[mid + 0x1])])
            right = mid
        else
            left = mid
        end
    end
    return right
end




# Create an integer linear space between start and stop on demand
struct IntLinSpace{T <: Integer}
    start::T
    stop::T
    length::T
end

function IntLinSpace(start::Integer, stop::Integer, length::Integer)
    start <= stop || throw(ArgumentError("`start` must be <= `stop`"))
    length >= 2 || throw(ArgumentError("`length` must be >= 2"))

    IntLinSpace{typeof(start)}(start, stop, length)
end

Base.IndexStyle(::IntLinSpace) = IndexLinear()
Base.length(ils::IntLinSpace) = ils.length

Base.firstindex(::IntLinSpace) = 1
Base.lastindex(ils::IntLinSpace) = ils.length

function Base.getindex(ils::IntLinSpace, i)
    @boundscheck 1 <= i <= ils.length || throw(BoundsError(ils, i))

    if i == 1
        ils.start
    elseif i == length
        ils.stop
    else
        ils.start + div((i - 1) * (ils.stop - ils.start), ils.length - 1, RoundUp)
    end
end
