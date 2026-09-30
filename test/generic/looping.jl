
@testset "foreachindex" begin
    Random.seed!(0)

    # CPU
    if AK._runs_threads(BACKEND)
        x = zeros(Int, 1000)
        AK.foreachindex(x) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; max_tasks=1, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; max_tasks=10, min_elems=1) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

        x = zeros(Int, 1000)
        AK.foreachindex(x; max_tasks=10, min_elems=10) do i
            x[i] = i
        end
        @test all(x .== 1:length(x))

    # GPU
    else
        x = array_from_host(zeros(Int, 10_000))
        f1(x) = AK.foreachindex(x) do i     # This must be inside a function to have a known type!
            x[i] = i
        end
        f1(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))

        x = array_from_host(zeros(Int, 10_000))
        f2(x) = AK.foreachindex(x; block_size=64) do i
            x[i] = i
        end
        f2(x)
        xh = Array(x)
        @test all(xh .== 1:length(xh))
    end

    # A range does not determine the backend: pass the backend of the arrays the loop accesses
    x = array_from_host(zeros(Int, 1000))
    f3(x) = AK.foreachindex(1:500; backend=AK.get_backend(x)) do i
        x[i] = i
    end
    f3(x)
    @test Array(x) == [1:500; zeros(Int, 500)]
    # ... without one, it runs on the host
    y = zeros(Int, 10)
    AK.foreachindex(i -> (y[i] = i), 1:10)
    @test y == 1:10

    # Invalid launch settings, on every backend
    for kw in ((block_size=0,), (max_tasks=0,), (min_elems=0,))
        @test_throws ArgumentError AK.foreachindex(i -> nothing, x; kw...)
    end
    @test_throws TypeError AK.foreachindex(i -> nothing, x; backend=:gpu)
    @test_throws MethodError AK.foreachindex(i -> nothing, x; prefer_threads=true)
    @test_throws MethodError AK.foreachindex(i -> nothing, x, AK.get_backend(x))
end


@testset "foraxes" begin
    Random.seed!(0)

    f1(x; kwargs...) = AK.foraxes(x, 1; kwargs...) do i
        for j in axes(x, 2)
            x[i, j] = i + j
        end
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f1(x)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    x = array_from_host(zeros(UInt32, 10, 1000))
    f1(x; max_tasks=2, min_elems=100, block_size=64)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    f2(x; kwargs...) = AK.foraxes(x, 2; kwargs...) do j
        for i in axes(x, 1)
            x[i, j] = i + j
        end
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f2(x)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    x = array_from_host(zeros(UInt32, 10, 1000))
    f2(x; max_tasks=2, min_elems=100, block_size=64)
    xh = Array(x)
    @test all(xh .== (1:10) .+ (1:1000)')

    # dims are nothing, behaving like foreachindex
    f3(x; kwargs...) = AK.foraxes(x, nothing; kwargs...) do i
        x[i] = i
    end

    x = array_from_host(zeros(Int, 10, 1000))
    f3(x)
    xh = Array(x)
    @test all(xh[:] .== 1:length(x))
end
