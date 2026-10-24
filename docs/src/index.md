![Logo](./assets/logo.png)

Parallel algorithm building blocks for the Julia ecosystem, targeting multithreaded CPUs, and GPUs via Nvidia CUDA, AMD ROCm, Intel oneAPI, Apple Metal and OpenCL (and any future backends added to the [JuliaGPU](https://juliagpu.org/) organisation), from one [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) codebase.

AcceleratedKernels.jl is a library of parallel primitives in the style of CUB: sorting, reductions,
prefix scans, stream compaction (`findall`), binary searches, `any`/`all`, `map`, `reverse` and
general loops. The functions take Base's names and arguments but follow a contract of their own,
modelled on CUB's (see [Differences from Base](@ref)). The sorts, reductions, scans, `findall` and
`any`/`all` take an explicit algorithm with its settings, and a reusable workspace for their
scratch memory. AcceleratedKernels does not try to
reproduce Base: packages such as GPUArrays.jl build Base's functions on top of it, and code that
needs control over the algorithm or its memory calls it directly.

---


## What's Different?
```@example
Main.DocHelpers.readme_section("## 1. What's Different?") # hide
```

---

## Status
```@example
Main.DocHelpers.readme_section("## 2. Status") # hide
```

---

## License
AcceleratedKernels.jl is MIT-licensed. Enjoy.
