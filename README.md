[![AcceleratedKernels.jl](https://github.com/juliagpu/AcceleratedKernels.jl/blob/main/docs/src/assets/logo.png?raw=true)](https://juliagpu.github.io/AcceleratedKernels.jl)

*"We need more speed" - Lightning McQueen or Scarface, I don't know*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagpu.github.io/AcceleratedKernels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagpu.github.io/AcceleratedKernels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Parallel algorithm building blocks for the Julia ecosystem, targeting multithreaded CPUs, and GPUs via Intel oneAPI, AMD ROCm, Apple Metal and Nvidia CUDA (and any future backends added to the [JuliaGPU](https://juliagpu.org/) organisation) from a unified KernelAbstractions.jl codebase.


### A Uniform API, Everywhere
Offering standard library algorithms (e.g., `sort`, `mapreduce`, `accumulate`), higher-order functions (e.g., `sum`, `cumprod`, `any`), and cross-architecture custom loops (`foreachindex`, `foraxes`), AcceleratedKernels.jl lets you write high-performance code once and run it on all supported architectures — no separate or special-cased kernels needed. It’s the classic “write once, run everywhere” principle, but supercharged for modern parallel CPU and GPU computing.


<table>

<tr>
<th>AcceleratedKernels Backend</th>
<th>Julia Version</th>
<th>CI Status</th>
</tr>

<tr>
<td>

CPU Single- and Multi-Threaded

</td>
<td>

Julia LTS, Stable, Pre-Release

x86, x64, aarch64

Windows, Ubuntu, MacOS

</td>
<td>

[![CI-CPU](https://github.com/juliagpu/AcceleratedKernels.jl/actions/workflows/CI-CPU.yml/badge.svg)](https://github.com/juliagpu/AcceleratedKernels.jl/actions/workflows/CI-CPU.yml)

</td>
</tr>

<tr>
<td rowspan=2>

[CUDA](https://github.com/JuliaGPU/CUDA.jl)

</td>
<td>

Julia v1.10

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=CUDA%20-%20Julia%20v1.10)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td>

Julia v1.11

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=CUDA%20-%20Julia%20v1.11)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td rowspan=2>

[AMDGPU](https://github.com/JuliaGPU/AMDGPU.jl)

</td>
<td>

Julia v1.10

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=AMDGPU%20-%20Julia%20v1.10)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td>

Julia v1.11

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=AMDGPU%20-%20Julia%20v1.11)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td rowspan=2>

[oneAPI](https://github.com/JuliaGPU/oneAPI.jl)

</td>
<td>

Julia v1.10

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=oneAPI%20-%20Julia%20v1.10)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td>

Julia v1.11

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=oneAPI%20-%20Julia%20v1.11)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td rowspan=2>

[Metal](https://github.com/JuliaGPU/Metal.jl)

</td>
<td>

Julia v1.10

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=Metal%20-%20Julia%20v1.10)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

<tr>
<td>

Julia v1.11

</td>
<td>

[![Build status](https://badge.buildkite.com/5b8c747451b382a6b1ad0a1b566d565bc851fc59515792c62e.svg?step=Metal%20-%20Julia%20v1.11)](https://buildkite.com/julialang/acceleratedkernels-dot-jl)

</td>
</tr>

</table>


- [1. What's Different?](#1-whats-different)
- [2. Status](#2-status)
- [3. Benchmarks](#3-benchmarks)
- [4. Functions Implemented](#4-functions-implemented)
- [5. API and Examples](#5-api-and-examples)
- [6. Custom Structs](#6-custom-structs)
- [7. Testing](#7-testing)
- [8. Issues and Debugging](#8-issues-and-debugging)
- [9. Roadmap / Future Plans](#9-roadmap--future-plans)
- [10. References](#10-references)
- [11. Acknowledgements](#11-acknowledgements)
- [12. License](#12-license)



## 1. What's Different?
As far as I am aware, this is the first cross-architecture parallel standard library *from a unified codebase* - that is, the code is written as [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) backend-agnostic kernels, which are then **transpiled** to a GPU backend; that means we benefit from all the optimisations available on the native platform and official compiler stacks. For example, unlike open standards like OpenCL that require GPU vendors to implement that API for their hardware, we target the existing official compilers. And while performance-portability libraries like [Kokkos](https://github.com/kokkos/kokkos) and [RAJA](https://github.com/LLNL/RAJA) are powerful for large C++ codebases, they require US National Lab-level development and maintenance efforts to effectively forward calls from a single API to other OpenMP, CUDA Thrust, ROCm rocThrust, oneAPI DPC++ libraries developed separately.

Again, this is only possible because of the unique Julia compilation model, the [JuliaGPU](https://juliagpu.org/) organisation work for reusable GPU backend infrastructure, and especially the [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) backend-agnostic kernel language. Thank you.


## 2. Status
The AcceleratedKernels.jl GPU `sort` and `accumulate` implementations were adopted as the official [AMDGPU algorithms](https://github.com/JuliaGPU/AMDGPU.jl/pull/688)! The API is starting to stabilise; it follows the Julia standard library fairly closely - and additionally exposing all temporary arrays for memory reuse. For any new ideas / requests, please join the conversation on [Julia Discourse](https://discourse.julialang.org/t/ann-acceleratedkernels-jl-cross-architecture-parallel-algorithms-for-julias-gpu-backends/119698/16) or post [an issue](https://github.com/juliagpu/AcceleratedKernels.jl/issues).

We have an extensive randomised test suite that we run on the CPU (single- and multi-threaded) backend on Windows, Ubuntu and MacOS for Julia LTS, Stable, and Pre-Release, plus the CUDA, AMDGPU, oneAPI and Metal backends on the [JuliaGPU buildkite](https://github.com/JuliaGPU/buildkite) - the exact same tests are run on all architectures to ensure uniform interfaces.

AcceleratedKernels.jl is also a fundamental building block of applications developed at [EvoPhase](https://evophase.co.uk/), so it will see continuous heavy use with industry backing. Long-term stability, performance improvements and support are priorities for us.


## 3. Benchmarks
Some arithmetic-heavy benchmarks are given below - see [this repository](https://github.com/anicusan/AcceleratedKernels-Benchmark) for the code; our paper will be linked here upon publishing with a full analysis.

![Arithmetic benchmark](https://github.com/anicusan/AcceleratedKernels-Benchmark/blob/main/ArithmeticBenchmark/ArithmeticBenchmarkTable.png?raw=true)

See `prototype/sort_benchmark.jl` for a small-scale sorting benchmark code and `prototype/thrust_sort` for the Nvidia Thrust wrapper. The results below are from a system with Linux 6.6.30-2-MANJARO, Intel Core i9-10885H CPU, Nvidia Quadro RTX 4000 with Max-Q Design GPU, Thrust 1.17.1-1, Julia Version 1.10.4.

![Sorting benchmark](https://github.com/juliagpu/AcceleratedKernels.jl/blob/main/docs/src/assets/sort_benchmark.png?raw=true)

As a first implementation in AcceleratedKernels.jl, we are on the same order of magnitude as Nvidia's official sorter (x3.48 slower), and an order of magnitude faster (x10.19) than the Julia Base CPU radix sort (which is already [one of the fastest](https://github.com/LilithHafner/InterLanguageSortingComparisons)).


The sorting algorithms can also be combined with [`MPISort.jl`](https://github.com/anicusan/MPISort.jl) for multi-*device* sorting - indeed, you can co-operatively sort using **both** your CPU and GPU! Or use 200 GPUs on the 52 nodes of [Baskerville HPC](https://www.baskerville.ac.uk/) to sort 538-855 GB of data per second (comparable with the highest figure reported in literature of [900 GB/s on 262,144 CPU cores](http://dx.doi.org/10.1145/2464996.2465442)):

![Sorting throughput](https://github.com/juliagpu/AcceleratedKernels.jl/blob/main/docs/src/assets/sort_throughput.png?raw=true)

Hardware stats for nerds [available here](https://docs.baskerville.ac.uk/system/). Full analysis will be linked here once our paper is published.


## 4. Functions Implemented

Below is an overview of the currently-implemented algorithms, along with some common names in other libraries for ease of finding / understanding / porting code - click on the function family to see the corresponding Manual entry.

If you need other algorithms in your work that may be of general use, please open an issue and we may implement it, help you implement it, or integrate existing code into AcceleratedKernels.jl.


| Function Family                               | AcceleratedKernels.jl Functions                  | Other Common Names                                        |
| --------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| [General Looping](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/foreachindex/) | `foreachindex`, `foraxes`                        | `Kokkos::parallel_for` `RAJA::forall` `thrust::transform` |
| [Mapping](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/map/) | `map` `map!`                                     | `thrust::transform`                                       |
| [Sorting](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/sort/) | `sort` `sort!`                                   | `sort` `sort_team` `stable_sort`                          |
|                                               | `sample_sort!` `sample_sortperm!`                |                                                           |
|                                               | `merge_sort` `merge_sort!`                       |                                                           |
|                                               | `merge_sort_by_key` `merge_sort_by_key!`         | `sort_team_by_key`                                        |
|                                               | `sortperm` `sortperm!`                           | `sort_permutation` `index_permutation`                    |
|                                               | `merge_sortperm` `merge_sortperm!`               |                                                           |
|                                               | `merge_sortperm_lowmem` `merge_sortperm_lowmem!` |                                                           |
| [Reduction](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/reduce/) | `reduce`                                         | `Kokkos:parallel_reduce` `fold` `aggregate`               |
| [MapReduce](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/mapreduce/) | `mapreduce`                                      | `transform_reduce` `fold`                                 |
| [Accumulation](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/accumulate/) | `accumulate` `accumulate!`                       | `prefix_sum` `thrust::scan` `cumsum`                      |
| [Binary Search](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/binarysearch/) | `searchsortedfirst` `searchsortedfirst!`         | `std::lower_bound`                                        |
|                                               | `searchsortedlast` `searchsortedlast!`           | `thrust::upper_bound`                                     |
| [Predicates](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/predicates/) | `all` `any`                                      |                                                           |
| [Arithmetics](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/arithmetics/) | `sum` `prod` `minimum` `maximum` `count` `cumsum` `cumprod`                                      |                                                           |



## 5. API and Examples

All publicly-exposed functions are [documented in the Manual](https://juliagpu.github.io/AcceleratedKernels.jl), along with examples to help you get started with AcceleratedKernels.jl. There are also plenty of examples and benchmarks in the `prototype` directory within slightly more involved scripts.


## 6. Custom Structs
As functions are compiled as/when used in Julia for the given argument types (for C++ people: kind of like everything being a template argument by default), we can use custom structs and functions defined outside AcceleratedKernels.jl, which will be inlined and optimised as if they were hardcoded within the library. Normal Julia functions and code can be used, without special annotations like `__device__`, `KOKKOS_LAMBDA` or wrapping them in classes with overloaded `operator()`.

As an example, let's compute the coordinate-wise minima of some points:
```julia
import AcceleratedKernels as AK
using Metal

struct Point
    x::Float32
    y::Float32
end

function compute_minima(points)
    AK.mapreduce(
        point -> (point.x, point.y),                    # Extract fields into tuple
        (a, b) -> (min(a[1], b[1]), min(a[2], b[2])),   # Keep each coordinate's minimum
        points,
        init=(typemax(Float32), typemax(Float32)),
        neutral=(typemax(Float32), typemax(Float32)),
    )
end

# Example output for Random.seed!(0):
#   minima = compute_minima(points) = (1.7966056f-5, 1.7797855f-6)
points = MtlArray([Point(rand(), rand()) for _ in 1:100_000])
@show minima = compute_minima(points)
```

Note that we did not have to explicitly type the function arguments in `compute_minima` - the types would be figured out when calling the function and compiled for the right backend automatically, e.g. CPU, oneAPI, ROCm, CUDA, Metal. Also, we used the standard Julia function `min`; it was not special-cased anywhere, it's just KernelAbstractions.jl inlining and compiling normal code, even from within the Julia.Base standard library.


## 7. Testing
If it ain't tested, it's broken. The `test/runtests.jl` suite does randomised correctness testing on all algorithms in the library. To test locally, execute:
```bash
$> julia -e 'import Pkg; Pkg.develop(path="path/to/AcceleratedKernels.jl"); Pkg.add("oneAPI")'
$> julia -e 'import Pkg; Pkg.test("AcceleratedKernels.jl", test_args=["--oneAPI"])'
```

Replace the `"--oneAPI"` with `"--CUDA"`, `"--AMDGPU"` or `"--Metal"` to test different backends, as available on your machine.

Leave out to test the CPU backend:
```bash
$> julia -e 'import Pkg; Pkg.test("AcceleratedKernels.jl")'
```


## 8. Issues and Debugging
As the compilation pipeline of GPU kernels is different to that of base Julia, error messages also look different - for example, where Julia would insert an exception when a variable name was not defined (e.g. we had a typo), a GPU kernel throwing exceptions cannot be compiled and instead you'll see some cascading errors like `"[...] compiling [...] resulted in invalid LLVM IR"` caused by `"Reason: unsupported use of an undefined name"` resulting in `"Reason: unsupported dynamic function invocation"`, etc.

Thankfully, there are only about 3 types of such error messages and they're not that scary when you look into them. See the Manual section on [debugging](https://juliagpu.github.io/AcceleratedKernels.jl/dev/debugging/) for examples and explanations.

For other library-related problems, feel free to post a GitHub issue. For help implementing new code, or just advice, you can also use the [Julia Discourse](https://discourse.julialang.org/c/domain/gpu/11) forum, the community is incredibly helpful.


## 9. Roadmap / Future Plans
Help is very welcome for any of the below:
- Automated optimisation / tuning of e.g. `block_size` for a given input; can be made algorithm-agnostic.
  - Maybe some thing like `AK.@tune reduce(f, src, init=init, block_size=$block_size) block_size=(64, 128, 256, 512, 1024)`. Macro wizards help!
  - Or make it general like:
  ```julia
  AK.@tune begin
      reduce(f, src, init=init,
             block_size=$block_size,
             switch_below=$switch_below)
      block_size=(64, 128, 256, 512, 1024)
      switch_below=(1, 10, 100, 1000, 10000)
  end
  ```
- Any way to expose the warp-size from the backends? Would be useful in reductions.
- Add a performance regressions runner.
- **Other ideas?** Post an issue, or open a discussion on the Julia Discourse.


## 10. References
This library is built on the unique Julia infrastructure for transpiling code to GPU backends, and years spent developing the [JuliaGPU](https://juliagpu.org/) ecosystem that make it a joy to use. In particular, credit should go to the following people and work:
- The Julia language design, which made code manipulation and generation a first class citizen: Bezanson J, Edelman A, Karpinski S, Shah VB. Julia: A fresh approach to numerical computing. SIAM review. 2017.
- The GPU compiler infrastructure built on top of Julia's unique compilation model: Besard T, Foket C, De Sutter B. Effective extensible programming: unleashing Julia on GPUs. IEEE Transactions on Parallel and Distributed Systems. 2018.
- The KernelAbstractions.jl library with its unique backend-agnostic compilation: Churavy V, Aluthge D, Wilcox LC, Schloss J, Byrne S, Waruszewski M, Samaroo J, Ramadhan A, Meredith SS, Bolewski J, Smirnov A. JuliaGPU/KernelAbstractions. jl: v0.8.3.
- For distributed applications, the MPI.jl library which makes integrating GPU codes with multi-node communication so easy: Byrne S, Wilcox LC, Churavy V. MPI. jl: Julia bindings for the Message Passing Interface. InProceedings of the JuliaCon Conferences 2021.

If you use AcceleratedKernels.jl in publications, please cite the works above.

While the algorithms themselves were implemented anew, multiple existing libraries and resources were useful; in no particular order:
- Kokkos: https://github.com/kokkos/kokkos
- RAJA: https://github.com/LLNL/RAJA
- Thrust / CUDA C++ Core Libraries: https://github.com/nvidia/cccl
- ThrustRTC: https://github.com/fynv/ThrustRTC
- Optimizing parallel reduction in CUDA: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- Parallel prefix sum (scan) with CUDA: https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf
- Parallel prefix sum (scan) with CUDA: https://github.com/mattdean1/cuda
- rocThrust: https://github.com/ROCm/rocThrust
- FidelityFX: https://github.com/GPUOpen-Effects/FidelityFX
- Intel oneAPI DPC++ library: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-library.html
- Metal performance shaders: https://developer.apple.com/documentation/metalperformanceshaders


## 11. Acknowledgements
Designed and built by [Andrei-Leonard Nicusan](https://github.com/anicusan), maintained with [contributors](https://github.com/juliagpu/AcceleratedKernels.jl/graphs/contributors).

Much of this work was possible because of the fantastic HPC resources at the University of Birmingham and the Birmingham Environment for Academic Research, which gave us free on-demand access to thousands of CPUs and GPUs that we experimented on, and the support teams we nagged. In particular, thank you to Kit Windows-Yule and Andrew Morris on the BlueBEAR and Baskerville T2 supercomputers' leadership, and Simon Branford, Simon Hartley, James Allsopp and James Carpenter for computing support.


## 12. License
AcceleratedKernels.jl is MIT-licensed. Enjoy.
