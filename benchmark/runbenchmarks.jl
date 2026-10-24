import AcceleratedKernels as AK
using KernelAbstractions
using GPUArrays

using BenchmarkTools
using StableRNGs
rng = StableRNG(123)

# parse command line args
BACKENDS = ["--CUDA", "--oneAPI", "--AMDGPU", "--Metal", "--OpenCL", "--CPU"]
b_opt_idx = in.(ARGS, Ref(BACKENDS))

if !@isdefined(backend_arg)
    backend_arg = if sum(b_opt_idx) == 0
        "--CPU"
    elseif sum(b_opt_idx) == 1
        only(ARGS[b_opt_idx])
    else
        throw(ArgumentError("More than one backend provided. Please retry with only one of $BACKENDS"))
    end
end
backend_arg in BACKENDS || throw(ArgumentError("\"$backend_arg\" is not a valid backend."))

other_args = ARGS[.!b_opt_idx]

# Files to ignore by default. Includes non-benchmark files and
#  backends can add incompatible benchmarks to this list
noinclude = ["runbenchmarks.jl"]

if backend_arg == "--CUDA"
    using CUDA
    CUDA.versioninfo()
    const ArrayType = CuArray
    macro sb(ex...)
        quote
            (CUDA.@sync blocking=true $(esc.(ex)...))
        end
    end
elseif backend_arg == "--oneAPI"
    using oneAPI
    oneAPI.versioninfo()
    const ArrayType = oneArray
    macro sb(ex...)
        quote
            oneAPI.@sync($(esc.(ex)...))
        end
    end
elseif backend_arg == "--AMDGPU"
    using AMDGPU
    AMDGPU.versioninfo()
    const ArrayType = ROCArray
    macro sb(ex...)
        quote
            AMDGPU.@sync($(esc.(ex)...))
        end
    end
elseif backend_arg == "--Metal"
    using Metal;
    Metal.versioninfo()
    const ArrayType = MtlArray
    macro sb(ex...)
        quote
            Metal.@sync($(esc.(ex)...))
        end
    end
    append!(noinclude, ["sort.jl", "sortperm.jl"])
elseif backend_arg == "--OpenCL"
    using OpenCL
    OpenCL.versioninfo()
    const ArrayType = CLArray
    macro sb(ex...) # Not sure how to sync
        quote
            $(esc.(ex)...)
        end
    end
elseif backend_arg == "--CPU"
    # Otherwise do CPU tests
    using InteractiveUtils
    InteractiveUtils.versioninfo()
    const ArrayType = Array
    macro sb(ex...)
        quote
            $(esc.(ex)...)
        end
    end
end


if backend_arg == "--CUDA"
    function reclaim_mem()
        GC.gc(true)
        CUDA.reclaim()
    end
else
    function reclaim_mem()
        GC.gc(true)
        GC.gc(true)
        GC.gc(true)
    end
end

# Select benchmarks to run
benches = filter(x -> endswith(x, ".jl") && x ∉ noinclude, Base.readdir())
if !isempty(other_args)
    benches = filter(x -> any(startswith.(Ref(x), other_args)), benches)
end

SUITE = BenchmarkGroup()
for b in benches
    include(b)
end

@info "Preparing benchmarks"
warmup(SUITE; verbose=false)
tune!(SUITE)

reclaim_mem()

@info "Running benchmarks"
results = run(SUITE, verbose=true)

BenchmarkTools.save("benchmarkresults.json", median(results))
