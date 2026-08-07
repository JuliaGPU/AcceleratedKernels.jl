import AcceleratedKernels as AK
using InteractiveUtils
using ParallelTestRunner

@info "Julia information:\n" * sprint(InteractiveUtils.versioninfo)

const init_code = quote
    import AcceleratedKernels as AK
    using KernelAbstractions
    using Test
    using Random
end

# Discover root-level tests (aqua.jl, partition.jl) and generic tests
const testsuite = find_tests(@__DIR__)
delete!(testsuite, "promote")
const generic_tests = find_tests(joinpath(@__DIR__, "generic"))

args = parse_args(ARGS; custom=["cpu-ka", "cpu"])

# GPU back-ends are tested when present in the test environment, which requires having
# explicitly promoted them from weak to regular dependencies (see test/promote.jl).
in_test_env(pkg) = Base.identify_package(pkg) !== nothing

# Common helper code appended to every backend setup
const _array_from_host_code = quote
    global array_from_host
    array_from_host(h_arr::AbstractArray, dtype=nothing) = array_from_host(BACKEND, h_arr, dtype)
    function array_from_host(backend, h_arr::AbstractArray, dtype=nothing)
        d_arr = d_arr = if prefer_threads # Don't use KA zeros if not using KA algorithms
            zeros(isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
        else
            KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
        end
        copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))
        d_arr
    end
end

# Build list of active backends, each with setup code
backends = Pair{String, Expr}[]


if in_test_env("CUDACore")
    using CUDACore, CUDATools
    @assert CUDACore.functional()
    @info "CUDACore information:\n" * sprint(CUDATools.versioninfo)
    push!(backends, "cuda" => quote
        using CUDACore
        global BACKEND = CUDABackend()
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = true
        $_array_from_host_code
    end)
end

if in_test_env("AMDGPU")
    using AMDGPU
    @assert AMDGPU.functional()
    println("AMDGPU information:")
    AMDGPU.versioninfo()
    push!(backends, "amdgpu" => quote
        using AMDGPU
        global BACKEND = ROCBackend()
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = true
        $_array_from_host_code
    end)
end

if in_test_env("Metal")
    using Metal
    @assert Metal.functional()
    @info "Metal information:\n" * sprint(Metal.versioninfo)
    push!(backends, "metal" => quote
        using Metal
        global BACKEND = MetalBackend()
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = false
        $_array_from_host_code
    end)
end

if in_test_env("oneAPI")
    using oneAPI
    @assert oneAPI.functional()
    @info "oneAPI information:\n" * sprint(oneAPI.versioninfo)
    push!(backends, "oneapi" => quote
        using oneAPI
        global BACKEND = oneAPIBackend()
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = false
        $_array_from_host_code
    end)
end

if in_test_env("OpenCL")
    using pocl_jll, OpenCL
    @assert !isempty(OpenCL.cl.platforms())
    @info "OpenCL information:\n" * sprint(OpenCL.versioninfo)
    push!(backends, "opencl" => quote
        using pocl_jll
        using OpenCL
        global BACKEND = OpenCLBackend()
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = false
        $_array_from_host_code
    end)
end

# cpu-ka only when --cpu-ka flag passed
if args.custom["cpu-ka"] !== nothing
    push!(backends, "cpu-ka" => quote
        global BACKEND = get_backend([])
        global prefer_threads = false   # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = false
        $_array_from_host_code
    end)
end

# CPU runs if no backend selected or if explicitly specified
if args.custom["cpu"] !== nothing || isempty(backends)
    push!(backends, "cpu" => quote
        global BACKEND = get_backend([])
        global prefer_threads = true    # Also used to determine whether to run the CPU or GPU tests
        global TEST_DL = false
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
