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

# The root-level tests run once; the generic ones (test/generic) once per backend. promote.jl is a
# script, and setup.jl the generic tests' setup.
const testsuite = find_tests(@__DIR__)
delete!(testsuite, "promote")
delete!(testsuite, "setup")
const generic_tests = find_tests(joinpath(@__DIR__, "generic"))

args = parse_args(ARGS; custom=["cpu-ka", "cpu"])

# GPU back-ends are tested when present in the test environment, which requires having
# explicitly promoted them from weak to regular dependencies (see test/promote.jl).
in_test_env(pkg) = Base.identify_package(pkg) !== nothing

# The backends under test, each with the code that sets up a generic test file for it: the
# globals the tests use, then test/setup.jl. `BACKEND` is the backend, `MAX_BLOCK_SIZE` its
# largest block, `TEST_KERNELS` whether AK's kernels run (else the threaded algorithms on host
# arrays), and `HOST_KERNELS` whether those kernels run on the host backend. `load` and the
# backend and block size are expressions, evaluated where the tests run.
backends = Pair{String, Expr}[]
function add_backend!(name, load, backend, max_block_size; kernels=true, host_kernels=false)
    push!(backends, name => quote
        $load
        global BACKEND = $backend
        global MAX_BLOCK_SIZE = Int($max_block_size)
        global TEST_KERNELS = $kernels
        global HOST_KERNELS = $host_kernels
        include($(joinpath(@__DIR__, "setup.jl")))
    end)
end

if in_test_env("CUDACore")
    using CUDACore, CUDATools
    @assert CUDACore.functional()
    @info "CUDACore information:\n" * sprint(CUDATools.versioninfo)
    add_backend!("cuda", :(using CUDACore), :(CUDABackend()),
                 :(CUDACore.attribute(CUDACore.device(),
                                      CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)))
end

if in_test_env("AMDGPU")
    using AMDGPU
    @assert AMDGPU.functional()
    println("AMDGPU information:")
    AMDGPU.versioninfo()
    add_backend!("amdgpu", :(using AMDGPU), :(ROCBackend()), 1024)
end

if in_test_env("Metal")
    using Metal
    @assert Metal.functional()
    @info "Metal information:\n" * sprint(Metal.versioninfo)
    add_backend!("metal", :(using Metal), :(MetalBackend()),
                 :(Metal.device().maxThreadsPerThreadgroup.width))
end

if in_test_env("oneAPI")
    using oneAPI
    @assert oneAPI.functional()
    @info "oneAPI information:\n" * sprint(oneAPI.versioninfo)
    add_backend!("oneapi", :(using oneAPI), :(oneAPIBackend()),
                 :(oneAPI.oneL0.compute_properties(oneAPI.device()).maxTotalGroupSize))
end

if in_test_env("OpenCL")
    using pocl_jll, OpenCL
    @assert !isempty(OpenCL.cl.platforms())
    @info "OpenCL information:\n" * sprint(OpenCL.versioninfo)
    add_backend!("opencl", :(using pocl_jll, OpenCL), :(OpenCLBackend()),
                 :(OpenCL.cl.device().max_work_group_size))
end

# AK's kernels on the host backend, only with --cpu-ka: KernelAbstractions 0.10 runs them (on
# PoCL), 0.9 does not
if args.custom["cpu-ka"] !== nothing
    AK.runs_kernels(AK.HOST_BACKEND) ||
        error("--cpu-ka needs KernelAbstractions 0.10, whose host backend runs AK's kernels")
    add_backend!("cpu-ka", nothing, :(get_backend([])), 1024; host_kernels=true)
end

# The threaded algorithms, if no other backend is selected or with --cpu
if args.custom["cpu"] !== nothing || isempty(backends)
    add_backend!("cpu", nothing, :(get_backend([])), 1024; kernels=false)
end

for (backend_name, setup) in backends
    for (test_name, test_body) in generic_tests
        testsuite["$backend_name/$test_name"] = quote
            $setup
            $test_body
        end
    end
end

# Filter tests by user-specified positional args; remove bare generic/ entries if no filter was specified
if filter_tests!(testsuite, args)
    filter!(((k,v),) -> !startswith(k, "generic/"), testsuite)
end

runtests(AK, args; init_code, testsuite)
