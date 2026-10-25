@testset "TaskPartitioner" begin
    # All tasks needed
    tp = AK.TaskPartitioner(10, 4, 1)
    @test tp.num_tasks == 4
    @test length(tp) == tp.num_tasks
    @test tp[1] === 1:3
    @test tp[2] === 4:6
    @test tp[3] === 7:8
    @test tp[4] === 9:10

    # Not all tasks needed
    tp = AK.TaskPartitioner(20, 6, 5)
    @test tp.num_tasks == 4
    @test length(tp) == tp.num_tasks
    @test tp[1] === 1:5
    @test tp[2] === 6:10
    @test tp[3] === 11:15
    @test tp[4] === 16:20
end


@testset "task_partition" begin
    Random.seed!(0)

    # Single-threaded
    x = zeros(Int, 1000)
    AK.task_partition(length(x), 1, 1) do irange
        for i in irange
            x[i] = i
        end
    end
    @test all(x .== 1:length(x))

    # Multi-threaded
    x = zeros(Int, 1000)
    tp = AK.TaskPartitioner(length(x), 10, 1)
    AK.task_partition(tp) do irange
        for i in irange
            x[i] = i
        end
    end
    @test all(x .== 1:length(x))

    # A partitioner with a single task
    x = zeros(Int, 10)
    AK.task_partition(AK.TaskPartitioner(length(x), 4, 8)) do irange
        x[irange] .= irange
    end
    @test x == 1:length(x)
end


@testset "itask_partition" begin
    Random.seed!(0)

    # Single-threaded
    x = zeros(Int, 1000)
    ix = zeros(Int, 1000)
    AK.itask_partition(length(x), 1, 1) do itask, irange
        for i in irange
            x[i] = i
            ix[i] = itask
        end
    end
    @test all(x .== 1:length(x))
    @test all(ix .== 1)

    # Multi-threaded
    x = zeros(Int, 1000)
    ix = zeros(Int, 1000)
    tp = AK.TaskPartitioner(length(x), 10, 1)
    AK.itask_partition(tp) do itask, irange
        for i in irange
            x[i] = i
            ix[i] = itask
        end
    end
    @test all(x .== 1:length(x))
    for i in 1:tp.num_tasks
        @test all(ix[tp[i]] .== i)
    end

    # A partitioner with a single task
    x = zeros(Int, 10)
    AK.itask_partition(AK.TaskPartitioner(length(x), 4, 8)) do itask, irange
        x[irange] .= itask
    end
    @test x == ones(Int, length(x))
end
