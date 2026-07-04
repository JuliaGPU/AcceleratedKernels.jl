
@testset "foreachindex" begin
    Random.seed!(0)

    # CPU
    if IS_CPU_BACKEND && prefer_threads
        x = zeros(Int, 1000)
        AK.foreachindex(x; prefer_threads) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; prefer_threads, max_tasks=1, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; prefer_threads, max_tasks=10, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; prefer_threads, max_tasks=10, min_elems=10) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

    # GPU
    else
        x = array_from_host(zeros(Int, 10_000))
        f1(x) = AK.foreachindex(x; prefer_threads) do i     # This must be inside a function to have a known type!
            x[i] = i
        end
        f1(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))

        x = array_from_host(zeros(Int, 10_000))
        f2(x) = AK.foreachindex(x; prefer_threads, block_size=64) do i
            x[i] = i
        end
        f2(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))
    end
end


@testset "foraxes" begin
    Random.seed!(0)

    f1(x; kwargs...) = AK.foraxes(x, 1; kwargs...) do i
        for j in axes(x, 2)
            x[i, j] = i + j
        end
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f1(x; prefer_threads)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    x = array_from_host(zeros(UInt32, 10, 1000))
    f1(x; prefer_threads, max_tasks=2, min_elems=100, block_size=64)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    f2(x; kwargs...) = AK.foraxes(x, 2; kwargs...) do j
        for i in axes(x, 1)
            x[i, j] = i + j
        end
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f2(x; prefer_threads)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    x = array_from_host(zeros(UInt32, 10, 1000))
    f2(x; prefer_threads, max_tasks=2, min_elems=100, block_size=64)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    # dims are nothing, behaving like foreachindex
    f3(x; kwargs...) = AK.foraxes(x, nothing; kwargs...) do i
        x[i] = i
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f3(x; prefer_threads)
    xh = Array(x)
    @test all(xh[:] .== 1:length(x))
end
