
@testset "MapReduce predicates" begin
    for _ in 1:100
        num_elems = rand(1:100_000)
        v = array_from_host(rand(Float32, num_elems))
        alg=AK.MapReduce(temp=similar(v, Bool), switch_below=100)
        @test AK.any(x->x<0, v; prefer_threads, alg) === false
        @test AK.any(x->x<1, v; prefer_threads, alg) === true
        @test AK.all(x->x<1, v; prefer_threads, alg) === true
        @test AK.all(x->x<0, v; prefer_threads, alg) === false
    end
end
