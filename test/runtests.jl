import AcceleratedKernels as AK
using ParallelTestRunner

const init_code = quote
    import AcceleratedKernels as AK
    using KernelAbstractions
    using Test
    using Random
end

# Discover root-level tests (aqua.jl, partition.jl) and generic tests
const testsuite = find_tests(@__DIR__)
const generic_tests = find_tests(joinpath(@__DIR__, "generic"))

# Parse args with lowercase hyphenated backend flags
args = parse_args(ARGS; custom=["cuda", "amdgpu", "metal", "oneapi", "opencl", "cpu-ka"])

# Common helper code appended to every backend setup
const _array_from_host_code = quote
    global array_from_host
    array_from_host(h_arr::AbstractArray, dtype=nothing) = array_from_host(BACKEND, h_arr, dtype)
    function array_from_host(backend, h_arr::AbstractArray, dtype=nothing)
        d_arr = KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
        copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))
        d_arr
    end
end

# Build list of active backends, each with setup code
backends = Pair{String, Expr}[]

# CPU always active
using InteractiveUtils
@info "Julia information:\n" * sprint(InteractiveUtils.versioninfo)
push!(backends, "cpu" => quote
    global BACKEND = get_backend([])
    global IS_CPU_BACKEND = true
    global prefer_threads = true
    global TEST_DL = Ref{Bool}(false)
    $_array_from_host_code
end)

# cpu-ka only when --cpu-ka flag passed
if args.custom["cpu-ka"] !== nothing
    push!(backends, "cpu-ka" => quote
        global BACKEND = get_backend([])
        global IS_CPU_BACKEND = true
        global prefer_threads = false
        global TEST_DL = Ref{Bool}(false)
        $_array_from_host_code
    end)
end

# GPU backends are only tested when explicitly requested via a CLI flag, in which case
# they are expected to be functional: load or initialization failures propagate.

if args.custom["cuda"] !== nothing
    using CUDA
    @assert CUDA.functional()
    @info "CUDA information:\n" * sprint(CUDA.versioninfo)
    push!(backends, "cuda" => quote
        using CUDA
        global BACKEND = CUDABackend()
        global IS_CPU_BACKEND = false
        global prefer_threads = true
        global TEST_DL = Ref{Bool}(true)
        $_array_from_host_code
    end)
end

if args.custom["amdgpu"] !== nothing
    using AMDGPU
    @assert AMDGPU.functional()
    @info "AMDGPU information:\n" * sprint(AMDGPU.versioninfo)
    push!(backends, "amdgpu" => quote
        using AMDGPU
        global BACKEND = ROCBackend()
        global IS_CPU_BACKEND = false
        global prefer_threads = true
        global TEST_DL = Ref{Bool}(true)
        $_array_from_host_code
    end)
end

if args.custom["metal"] !== nothing
    using Metal
    @assert Metal.functional()
    @info "Metal information:\n" * sprint(Metal.versioninfo)
    push!(backends, "metal" => quote
        using Metal
        global BACKEND = MetalBackend()
        global IS_CPU_BACKEND = false
        global prefer_threads = true
        global TEST_DL = Ref{Bool}(false)
        $_array_from_host_code
    end)
end

if args.custom["oneapi"] !== nothing
    using oneAPI
    @assert oneAPI.functional()
    @info "oneAPI information:\n" * sprint(oneAPI.versioninfo)
    push!(backends, "oneapi" => quote
        using oneAPI
        global BACKEND = oneAPIBackend()
        global IS_CPU_BACKEND = false
        global prefer_threads = true
        global TEST_DL = Ref{Bool}(false)
        $_array_from_host_code
    end)
end

if args.custom["opencl"] !== nothing
    using pocl_jll, OpenCL
    @assert !isempty(OpenCL.cl.platforms())
    @info "OpenCL information:\n" * sprint(OpenCL.versioninfo)
    push!(backends, "opencl" => quote
        using pocl_jll
        using OpenCL
        global BACKEND = OpenCLBackend()
        global IS_CPU_BACKEND = false
        global prefer_threads = true
        global TEST_DL = Ref{Bool}(false)
        $_array_from_host_code
    end)
end

# Duplicate generic tests per active backend
for (backend_name, setup_code) in backends
    for (test_name, test_body) in generic_tests
        testsuite["$backend_name/$test_name"] = quote
            $setup_code
            $test_body
        end
    end
end

# Filter tests by user-specified positional args; remove bare generic/ entries if no filter was specified
if filter_tests!(testsuite, args)
    filter!(((k,v),) -> !startswith(k, "generic/"), testsuite)
end

runtests(AK, args; init_code, testsuite)
