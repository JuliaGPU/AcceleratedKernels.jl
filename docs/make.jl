using AcceleratedKernels
using Documenter


makedocs(;
    modules=[AcceleratedKernels],
    authors="Andrei-Leonard Nicusan <a.l.nicusan@gmail.com> and contributors",
    sitename="AcceleratedKernels.jl",
    format=Documenter.HTML(;
        canonical="https://juliagpu.github.io/KernelAbstractions.jl/",
        edit_link="main",
        assets=String[],
        sidebar_sitename=false,

        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Overview" => "index.md",
        "Benchmarks" => "benchmarks.md",
        "Performance Tips" => "performance.md",
        "Manual" =>[
            "Using Different Backends" => "api/using_backends.md",
            "General Loops" => "api/foreachindex.md",
            "Map" => "api/map.md",
            "Random Number Generation" => "api/rand.md",
            "Sorting" => "api/sort.md",
            "Reduce" => "api/reduce.md",
            "MapReduce" => "api/mapreduce.md",
            "Accumulate" => "api/accumulate.md",
            "Binary Search" => "api/binarysearch.md",
            "Predicates" => "api/predicates.md",
            "Arithmetics" => "api/arithmetics.md",
            "Custom Structs" => "api/custom_structs.md",
            "Task Partitioning" => "api/task_partition.md",
            "Utilities" => "api/utilities.md",
        ],
        "Testing" => "testing.md",
        "Debugging Kernels" => "debugging.md",
        "Roadmap" => "roadmap.md",
        "References" => "references.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/JuliaGPU/AcceleratedKernels.jl",
    devbranch="main",
)
