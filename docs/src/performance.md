## Performance Tips

If you just started using `AcceleratedKernels.jl`, see the Manual first for some examples.


### GPU Block Size and CPU Threads

All GPU functions allow you to specify a block size - this is often a power of two (mostly 64, 128, 256, 512); the optimum depends on the algorithm, input data and hardware - you can try the different values and `@time` or `@benchmark` them:
```julia
@time AK.foreachindex(f, itr_gpu, block_size=512)
```

Similarly, for performance on the CPU the overhead of spawning threads should be masked by processing more elements per thread (but there is no reason here to launch more threads than `Threads.nthreads()`, the number of threads Julia was started with); the optimum depends on how expensive `f` is - again, benchmarking is your friend:
```julia
@time AK.foreachindex(f, itr_cpu, max_tasks=16, min_elems=1000)
```


### Scratch Memory

As GPU memory is more expensive, every operation that needs scratch memory can take it from a
workspace, allocated once for a kind of call and reused (see [Scratch Memory](api/workspace.md)):
```julia
v = ROCArray(rand(Float32, 100_000))
ws = AK.workspace(AK.sort!, v)
AK.sort!(v; workspace=ws)
```



