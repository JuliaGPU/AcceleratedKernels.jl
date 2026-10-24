# Local memory that kernels with a tunable footprint may use: the minimum an OpenCL device must
# provide (`CL_DEVICE_LOCAL_MEM_SIZE`, full profile), and Metal's threadgroup memory limit.
const LOCAL_MEMORY_BUDGET = 32 * 1024

# The default of every `init` keyword: no initial value given. As in Base, this differs from
# `init=nothing`, which is an explicit initial value.
struct NoInit end

@inline unval(::Val{V}) where {V} = V
