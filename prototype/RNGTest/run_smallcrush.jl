using RNGTest

include("stream.jl")


const ALG = :philox
const SEED = 0x1234
const CHUNK = 100_000_000
const HOST_SCRATCH = Vector{UInt64}(undef, CHUNK)


stream = AKUInt64Stream(
    HOST_SCRATCH;
    seed=SEED,
    alg=ALG,
    start_counter=UInt64(0),
)
gen = make_rngtest_generator!(stream)
genname = "AK_Vector_$(ALG)_seed$(SEED)"

println("Beginning SmallCrush...")

RNGTest.smallcrushTestU01(gen, genname)
