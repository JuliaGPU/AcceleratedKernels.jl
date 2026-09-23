# Sweep AcceleratedKernels' GPU sorting algorithms against each other on one device, to choose the
# per-device sorting defaults: when `BitonicSort` or `RadixSort` beats `MergeSort`, and which
# `block_size`/`items_per_thread` each algorithm should use.
#
# Usage, from an environment with AcceleratedKernels and the backend package:
#
#     julia --project=<env> benchmark/tune_sort.jl <backend> [--quick] [--csv=<file>] [--platform=<name>]
#
# where <backend> is one of cuda, amdgpu, metal, oneapi, opencl (the default OpenCL device, or the
# first device of the platform whose name contains `--platform`) or pocl (OpenCL on pocl_jll's CPU
# device). `--quick` measures fewer lengths and shorter; `--csv` also writes every measurement.
#
# Method. Every measurement restores the input from a pristine copy, synchronises, then times one
# sort end to end including the final synchronisation, and reports the median over repeated
# samples (at least 3, until a time budget is spent). The first run of every configuration also
# compiles it and checks that the result is sorted.
#
#   Phase A tunes each algorithm's `block_size`/`items_per_thread` separately for 4- and 8-byte
#   element types, on a few representative whole-array and per-slice sorts; the score of a
#   setting is the geometric mean of its time relative to today's default.
#   Phase B uses those settings to compare the algorithms over element types, whole-array lengths
#   and per-slice shapes (along `dims=1`, along a strided dimension, and through a view), and three
#   input distributions: full-range uniform, few distinct values, and a narrow key range (which
#   lets radix sort skip passes).
#
# From phase B the script derives, for 4- and 8-byte element types separately and conservatively
# over the element types of each size, their distributions and slice layouts:
#   - the largest length up to which `BitonicSort` is never slower than the alternatives, for
#     whole arrays and for slices separately;
#   - the smallest whole-array length from which `RadixSort` is never slower than `MergeSort`.
# and prints them as `SortTuning` literals, the per-device values AK's `sort_tuning` hook returns.

using Random
import AcceleratedKernels as AK
using KernelAbstractions
const KA = KernelAbstractions

const ARGS_BACKEND = isempty(ARGS) ? error("usage: tune_sort.jl <backend> [--quick] [--csv=<file>]") : lowercase(ARGS[1])
const QUICK = "--quick" in ARGS
const CSV_PATH = let a = findfirst(startswith("--csv="), ARGS)
    a === nothing ? nothing : split(ARGS[a], '=', limit=2)[2]
end
const PLATFORM = let a = findfirst(startswith("--platform="), ARGS)
    a === nothing ? nothing : split(ARGS[a], '=', limit=2)[2]
end

if ARGS_BACKEND == "cuda"
    using CUDA
    const BACKEND = CUDABackend()
    device_name() = CUDA.name(CUDA.device())
elseif ARGS_BACKEND == "amdgpu"
    using AMDGPU
    const BACKEND = ROCBackend()
    device_name() = string(AMDGPU.device())
elseif ARGS_BACKEND == "metal"
    using Metal
    const BACKEND = MetalBackend()
    device_name() = String(Metal.device().name)
elseif ARGS_BACKEND == "oneapi"
    using oneAPI
    const BACKEND = oneAPIBackend()
    device_name() = oneAPI.oneL0.properties(oneAPI.device()).name
elseif ARGS_BACKEND in ("opencl", "pocl")
    if ARGS_BACKEND == "pocl"
        using pocl_jll
    end
    using OpenCL
    let name = ARGS_BACKEND == "pocl" ? "Portable Computing Language" : PLATFORM
        if name !== nothing
            platform = only(filter(p -> occursin(name, p.name), cl.platforms()))
            cl.platform!(platform)
        end
    end
    const BACKEND = OpenCLBackend()
    device_name() = cl.device().name
else
    error("unknown backend $(ARGS_BACKEND)")
end

sync() = KA.synchronize(BACKEND)

function device_array(h::AbstractArray)
    d = KA.allocate(BACKEND, eltype(h), size(h))
    copyto!(d, h)
    d
end


# Inputs

const DISTRIBUTIONS = (:uniform, :few, :narrow)

function host_input(::Type{T}, dims, dist) where T
    n = prod(dims)
    h = if dist === :uniform
        T <: AbstractFloat ? rand(T, n) : rand(T, n)
    elseif dist === :few
        T.(rand(1:16, n))
    elseif dist === :narrow
        # Keys differ only in their 12 lowest bits, so radix sort skips its upper passes
        if T <: AbstractFloat
            one(T) .+ T.(rand(0:4095, n)) .* eps(one(T))
        else
            T(rand(T) >> 16) .+ T.(rand(0:4095, n))
        end
    end
    reshape(h, dims)
end

eltypes() = filter(T -> T !== Float64 || KA.supports_float64(BACKEND),
                   [UInt32, Int32, Float32, Int64, Float64, Int16])
radix_ok(::Type{T}) where T = T in (UInt32, Int32, Float32, UInt64, Int64, Float64)
sizeclass(::Type{T}) where T = sizeof(T) <= 4 ? 4 : 8


# Measurement

const MIN_SAMPLES = 3
const MAX_SAMPLES = QUICK ? 15 : 50
const BUDGET = QUICK ? 0.05 : 0.25     # seconds per configuration, beyond MIN_SAMPLES

# Sort `v` in place with `f!(v)`; restore it from `v0` before every sample. Returns the median
# time in seconds, or NaN if the configuration is not supported (an argument or launch error).
function measure(f!, v, v0; check=x -> true)
    try
        copyto!(v, v0); sync()
        f!(v); sync()
    catch err
        err isa InterruptException && rethrow()
        return NaN
    end
    check(v) || error("configuration produced an unsorted result")
    times = Float64[]
    t_end = time() + BUDGET
    while length(times) < MIN_SAMPLES || (length(times) < MAX_SAMPLES && time() < t_end)
        copyto!(v, v0); sync()
        t0 = time_ns()
        f!(v); sync()
        push!(times, (time_ns() - t0) / 1e9)
    end
    sort!(times)[cld(length(times), 2)]
end

slices_sorted(dims) = v -> begin
    h = Array(v)
    dims isa Colon ? issorted(vec(h)) : all(issorted, eachslice(h; dims=Tuple(d for d in 1:ndims(h) if d != dims)))
end

merge_sorter(bs; dims=:) = v -> AK.sort!(v; alg=AK.MergeSort(), block_size=bs, dims)
radix_sorter(bs, ipt) = v -> AK.sort!(v; alg=AK.RadixSort(block_size=bs, items_per_thread=ipt))
bitonic_sorter(bs, ipt; dims=:) = v -> AK.sort!(v; alg=AK.BitonicSort(block_size=bs, items_per_thread=ipt), dims)

const RECORDS = NamedTuple[]

function record!(phase, alg, T, shape, dist, bs, ipt, t)
    push!(RECORDS, (; phase, alg, T=string(T), shape, dist=string(dist), block_size=bs,
                    items_per_thread=ipt, time_us=t * 1e6))
    t
end


# Phase A: tunables

# Today's defaults, the baseline of the scores
const DEFAULTS = (merge=(256,), radix=(256, 2), bitonic=(256, 8))

function geomean_ratio(times, base)
    rs = [t / b for (t, b) in zip(times, base) if isfinite(t) && isfinite(b)]
    length(rs) == length(times) || return Inf      # unsupported somewhere: never pick it
    exp(sum(log, rs) / length(rs))
end

function tune(alg, T, cases, settings, mk)
    times = Dict{Any,Vector{Float64}}()
    for s in settings
        times[s] = map(cases) do (shape, dims)
            h = host_input(T, shape, :uniform)
            v0 = device_array(h); v = similar(v0)
            t = measure(mk(s..., dims), v, v0; check=slices_sorted(dims))
            record!("A", alg, T, (shape, dims), :uniform, s[1], length(s) > 1 ? s[2] : 0, t)
        end
    end
    base = times[getfield(DEFAULTS, alg)]
    scores = Dict(s => geomean_ratio(ts, base) for (s, ts) in times)
    best = argmin(s -> scores[s], collect(keys(scores)))
    best, scores
end

function phase_a()
    println("\n## Phase A: tunables (score = geometric mean of time relative to today's default)\n")
    flat(ks) = [((2^k,), :) for k in ks]
    ks_large = QUICK ? (16, 20) : (16, 20, 24)
    best = Dict{Tuple{Symbol,Int},Any}()
    for T in (UInt32, UInt64)
        c = sizeclass(T)
        merge_cases = [flat(ks_large); [((2^10, 2^20 ÷ 2^10), 1)]]
        best[(:merge, c)], sc = tune(:merge, T, merge_cases, [(bs,) for bs in (64, 128, 256, 512)],
                                     (bs, dims) -> merge_sorter(bs; dims))
        report_scores("MergeSort", T, sc)
        best[(:radix, c)], sc = tune(:radix, T, flat(ks_large),
                                     [(bs, ipt) for bs in (128, 256, 512) for ipt in (1, 2, 4, 8)],
                                     (bs, ipt, dims) -> radix_sorter(bs, ipt))
        report_scores("RadixSort", T, sc)
        bitonic_cases = [flat(QUICK ? (10, 16) : (10, 14, 18));
                         [((2^k, 2^20 ÷ 2^k), 1) for k in (QUICK ? (6, 12) : (6, 10, 13))]]
        best[(:bitonic, c)], sc = tune(:bitonic, T, bitonic_cases,
                                       [(bs, ipt) for bs in (64, 128, 256, 512) for ipt in (2, 4, 8, 16)],
                                       (bs, ipt, dims) -> bitonic_sorter(bs, ipt; dims))
        report_scores("BitonicSort", T, sc)
    end
    best
end

function report_scores(name, T, scores)
    ranked = sort!(collect(scores), by=last)
    println("$name, $T: ", join(("$(k) => $(round(v, digits=3))" for (k, v) in ranked[1:min(5, end)]), ", "),
            isempty(ranked) ? "" : "  (worst: $(ranked[end][1]) => $(round(ranked[end][2], digits=3)))")
end


# Phase B: crossovers

flat_lengths() = QUICK ? [2^k for k in 6:2:22] :
    sort!([[2^k for k in 6:26]; [3 * 2^(k - 1) for k in 7:24]])
slice_lengths() = QUICK ? [2^k for k in 4:2:14] : [2^k for k in 4:14]
slice_totals() = QUICK ? (2^20,) : (2^20, 2^24)

# Stop timing bitonic sort once it has been this much slower than the best for two lengths in a
# row; its O(n log² n) cost does not cross back.
const GIVE_UP = 4.0

function phase_b(best)
    flat_res = Dict{Any,Dict{Symbol,Float64}}()      # (T, dist, n) => alg => time
    println("\n## Phase B: whole-array sorts (median μs; — = not run)\n")
    for T in eltypes(), dist in DISTRIBUTIONS
        c = sizeclass(T)
        algs = radix_ok(T) ? (:merge, :radix, :bitonic) : (:merge, :bitonic)
        losing = Dict(a => 0 for a in algs)
        println("$T, $dist:")
        println("  ", rpad("n", 10), join((lpad(string(a), 12) for a in algs)))
        for n in flat_lengths()
            h = host_input(T, (n,), dist)
            v0 = device_array(h); v = similar(v0)
            r = Dict{Symbol,Float64}()
            for a in algs
                losing[a] >= 2 && (r[a] = NaN; continue)
                s = best[(a, c)]
                sorter = a === :merge ? merge_sorter(s...) : a === :radix ? radix_sorter(s...) : bitonic_sorter(s...)
                r[a] = record!("B", a, T, (n,), dist, s[1], length(s) > 1 ? s[2] : 0,
                               measure(sorter, v, v0; check=slices_sorted(:)))
            end
            tbest = minimum(t for t in values(r) if isfinite(t))
            if isfinite(r[:bitonic])
                losing[:bitonic] = r[:bitonic] > GIVE_UP * tbest ? losing[:bitonic] + 1 : 0
            end
            flat_res[(T, dist, n)] = r
            println("  ", rpad(n, 10), join((lpad(isfinite(r[a]) ? string(round(r[a] * 1e6, digits=1)) : "—", 12) for a in algs)))
        end
    end

    slice_res = Dict{Any,Dict{Symbol,Float64}}()     # (T, layout, total, len) => alg => time
    println("\n## Phase B: per-slice sorts, uniform input (median μs)\n")
    for T in eltypes(), total in slice_totals(), layout in (:dims1, :dims2, :view)
        c = sizeclass(T)
        println("$T, total $total, $layout:")
        println("  ", rpad("len × count", 18), lpad("merge", 12), lpad("bitonic", 12))
        for len in slice_lengths()
            count = total ÷ len
            if layout === :dims1
                shape, dims = (len, count), 1
            elseif layout === :dims2
                shape, dims = (count, len), 2
            else
                shape, dims = (len + 1, count), 1       # sorted through view(A, 1:len, :)
            end
            h = host_input(T, shape, :uniform)
            p0 = device_array(h); p = similar(p0)
            v0, v = layout === :view ? (view(p0, 1:len, :), view(p, 1:len, :)) : (p0, p)
            r = Dict{Symbol,Float64}()
            for a in (:merge, :bitonic)
                s = best[(a, c)]
                sorter = a === :merge ? merge_sorter(s...; dims) : bitonic_sorter(s...; dims)
                r[a] = record!("B", a, T, (layout, len, count), :uniform, s[1], length(s) > 1 ? s[2] : 0,
                               measure(sorter, v, v0; check=slices_sorted(dims)))
            end
            slice_res[(T, layout, total, len)] = r
            println("  ", rpad("$len × $count", 18), join((lpad(isfinite(r[a]) ? string(round(r[a] * 1e6, digits=1)) : "—", 12) for a in (:merge, :bitonic))))
        end
    end
    flat_res, slice_res
end


# Thresholds

# Largest length L such that at every measured length <= L, `alg` is no slower than every other
# algorithm measured there (0 if it loses at the smallest length).
function max_winning_len(res, key, lengths, alg)
    L = 0
    for n in lengths
        r = res[key(n)]
        t = r[alg]
        isfinite(t) && t <= minimum(x for x in values(r) if isfinite(x)) || break
        L = n
    end
    L
end

# Smallest length L such that at every measured length >= L, `alg` beats `other`
# (typemax(Int) if it does not win at the largest length).
function min_winning_len(res, key, lengths, alg, other)
    L = typemax(Int)
    for n in reverse(lengths)
        r = res[key(n)]
        isfinite(r[alg]) && (!isfinite(r[other]) || r[alg] <= r[other]) || break
        L = n
    end
    L
end

# Thresholds per element-size class (4 or 8 bytes), each conservative over the element types of
# that class, their distributions and slice layouts
function thresholds(flat_res, slice_res)
    println("\n## Crossovers\n")
    lens = flat_lengths()
    th = Dict(c => (bitonic_flat=typemax(Int), bitonic_slices=typemax(Int), radix_min=0) for c in (4, 8))
    for T in eltypes(), dist in DISTRIBUTIONS
        b = max_winning_len(flat_res, n -> (T, dist, n), lens, :bitonic)
        # The radix crossover is against merge sort only: `Auto` considers radix sort after it
        # has ruled out bitonic sort by length.
        r = radix_ok(T) ? min_winning_len(flat_res, n -> (T, dist, n), lens, :radix, :merge) : nothing
        println("whole array, $T, $dist: bitonic wins up to $b",
                r === nothing ? "" : ", radix beats merge from $(r == typemax(Int) ? "never" : r)")
        t = th[sizeclass(T)]
        th[sizeclass(T)] = (bitonic_flat=min(t.bitonic_flat, b), bitonic_slices=t.bitonic_slices,
                            radix_min=r === nothing ? t.radix_min : max(t.radix_min, r))
    end
    for T in eltypes(), total in slice_totals(), layout in (:dims1, :dims2, :view)
        b = max_winning_len(slice_res, len -> (T, layout, total, len), slice_lengths(), :bitonic)
        println("slices, $T, total $total, $layout: bitonic wins up to $b")
        t = th[sizeclass(T)]
        th[sizeclass(T)] = (bitonic_flat=t.bitonic_flat, bitonic_slices=min(t.bitonic_slices, b),
                            radix_min=t.radix_min)
    end
    th
end


function main()
    Random.seed!(0)
    println("# AcceleratedKernels sort tuning: $(device_name()) ($(ARGS_BACKEND)), $(QUICK ? "quick" : "full") sweep")
    println("Julia $(VERSION), AcceleratedKernels $(pkgversion(AK)), $(Sys.cpu_info()[1].model), $(Dates_now())")

    best = phase_a()
    flat_res, slice_res = phase_b(best)
    th = thresholds(flat_res, slice_res)

    fmt(x) = x == typemax(Int) ? "typemax(Int)" : string(x)
    println("\n## Result\n")
    println("""
    One `SortTuning` per element-size class. `bitonic_max_len` is the whole-array crossover and
    `bitonic_max_slice_len` the per-slice one; where they differ, `SortTuning` needs both fields.""")
    for c in (4, 8)
        m, r, b, t = best[(:merge, c)], best[(:radix, c)], best[(:bitonic, c)], th[c]
        println("""

        # $(device_name()), $(c)-byte eltypes, benchmark/tune_sort.jl$(QUICK ? " --quick" : ""), $(Dates_now())
        AK.SortTuning(
            bitonic_max_len = $(fmt(t.bitonic_flat)),
            bitonic_max_slice_len = $(fmt(t.bitonic_slices)),
            radix_min_len = $(fmt(t.radix_min)),
            merge_block_size = $(m[1]),
            radix_block_size = $(r[1]),
            radix_items_per_thread = $(r[2]),
            bitonic_block_size = $(b[1]),
            bitonic_items_per_thread = $(b[2]),
        )""")
    end

    if CSV_PATH !== nothing
        open(CSV_PATH, "w") do io
            println(io, join(keys(first(RECORDS)), ','))
            for r in RECORDS
                println(io, join((x isa Tuple ? "\"$(x)\"" : string(x) for x in values(r)), ','))
            end
        end
        println("\nWrote $(length(RECORDS)) measurements to $(CSV_PATH)")
    end
end

Dates_now() = string(Base.Libc.strftime("%Y-%m-%d", time()))

main()
