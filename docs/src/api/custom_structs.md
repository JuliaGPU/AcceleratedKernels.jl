### Custom Structs

```@example
import AcceleratedKernels as AK # hide
AK.DocHelpers.readme_section("## 6. Custom Structs") # hide
```

You can also use unmaterialised index ranges in GPU kernels - unmaterialised meaning you do not
need to waste memory creating a vector of indices, e.g.:

```julia
import AcceleratedKernels as AK
using CUDA

function complex_any(x, y)
    # Calling `any` on a normal Julia range, but running on x's backend
    AK.any(1:length(x); backend=AK.get_backend(x)) do i
        x[i] < 0 && y[i] > 0
    end
end

complex_any(CuArray(rand(Float32, 100)), CuArray(rand(Float32, 100)))
```

Note that you have to specify the `backend` explicitly in this case, as a range does not have a
backend per se - for example, when used in a GPU kernel, it only passes two numbers, the
`Base.UnitRange` start and stop, as saved in a basic struct, rather than a whole vector.
