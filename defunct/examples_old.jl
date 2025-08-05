using Revise

using DrawingInferences
using Distributions
using BenchmarkTools
using Random
using MacroTools


initialState = WeightedSampler(
    (rng) -> randn(rng),
    (x::Float64) -> 0.0,
    (x::Float64) -> 0.0
)

randomWalkKernel = WeightedSampler(
    (x_in::Float64, rng) -> x_in + randn(rng),
    (x_in::Float64, x_out::Float64) -> 0.0,
    (x_in::Float64, x_out::Float64) -> 0.0
)

function run_fk(fk)
    model = SMCModel(fk)
    N_particles = 2^10
    nthreads = 1 #Threads.nthreads()
    fullOutput = false
    essThreshold = 2.0

    smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

    @time smc!(model, smcio)
end

@time @model function randomWalkModel(T::Int64) # Does barely any computation
    x::Float64 ~ initialState()
    for i in 1:T
        x::Float64 ~ randomWalkKernel(x)
    end
end

@time fk = randomWalkModel(100) # Compiling takes a while (4s), could be optimized

@time model = SMCModel(fk) # fast 0.1s
N_particles = 2^10
nthreads = 1 #Threads.nthreads()
fullOutput = false
essThreshold = 2.0

@time smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)


@time smc!(model, smcio)
@btime smc!(model, smcio) # Also has a long compilation time (25s), but is somewhat independent of the particle number


# With Vector{Any} lookup tables in SMCModel, compilation takes 5s at the cost of introducing 5.6M allocations.

### Performance tradeoff (100 steps)

# Vector{Any} lookup tables in SMCModel:
# 1s compilation model creation + 1s compilation smc!
# 2^10 ~1k particles: 12.3ms 569k allocations
# 2^15 ~32k particles: 411ms 20M allocations
# 2^20 ~1M particles: 15s 600M allocations

#Tuple lookup tables in SMCModel:
# 0.5s model creation +  0.8s smc!
# 2^10 ~1k particles: 5ms 0 allocations
# 2^15 ~32k particles: 160ms 0 allocations
# 2^20 ~1M particles: 6s 0 allocations (5s compilation)

### Performance tradeoff (1_000 steps)

# Vector{Any} lookup tables in SMCModel:
# 4s model creation + 4s compilation smc!
# 2^10 ~1k particles: 250ms 5.6M allocations
# 2^15 ~32k particles: 7.4s 200M allocations
# 2^20 ~1M particles: 240s 6.4G allocations

#Tuple lookup tables in SMCModel:
# 4s model creation + 35s smc! compilation
# 2^10 ~1k particles: 61ms 0 allocations
# 2^15 ~32k particles: 1.9s 0 allocations 
# 2^20 ~1M particles: 70s 0 allocations

# Need better fk representation that can capture the structure of the model, e.g. repeated steps.