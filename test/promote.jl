# Promote a GPU back-end from a weak to a regular dependency of the test environment,
# so that Pkg.test installs it. Rewrites the TOML directly rather than using Pkg, which
# avoids a second resolution as well as Pkg.test sandbox issues on Julia 1.10.
#
# Usage: julia test/promote.jl <cuda|amdgpu|metal|oneapi|opencl>...

using TOML

const BACKEND_PACKAGES = Dict(
    "cuda"   => ["CUDACore", "CUDATools"],
    "amdgpu" => ["AMDGPU"],
    "metal"  => ["Metal"],
    "oneapi" => ["oneAPI"],
    "opencl" => ["OpenCL", "pocl_jll"],
)

const project_file = joinpath(@__DIR__, "Project.toml")

proj = TOML.parsefile(project_file)
for backend in ARGS
    pkgs = get(BACKEND_PACKAGES, lstrip(lowercase(backend), '-'), nothing)
    pkgs === nothing &&
        error("Unknown back-end '$backend'; expected one of: " *
              join(sort!(collect(keys(BACKEND_PACKAGES))), ", "))
    for pkg in pkgs
        proj["deps"][pkg] = pop!(proj["weakdeps"], pkg)
    end
    @info "Promoted $(join(pkgs, ", ")) to test dependencies"
end
open(io -> TOML.print(io, proj), project_file, "w")
