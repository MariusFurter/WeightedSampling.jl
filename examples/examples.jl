using Revise

using DrawingInferences
using Distributions
using BenchmarkTools
using Random
using MacroTools


initialState = WeightedSampler(
    (rng) -> rand(rng, Normal(0, 1)),
    (x::Float64) -> logpdf(Normal(0, 1), x),
    (x::Float64) -> 0.0
)

randomWalkKernel = WeightedSampler(
    (x_in::Float64, rng) -> rand(rng, Normal(x_in, 1)),
    (x_in::Float64, x_out::Float64) -> logpdf(Normal(x_in, 1), x_out),
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

@time @model function randomWalkModel(T::Int64)
    x::Float64 ~ initialState()
    for i in 1:T
        x::Float64 ~ randomWalkKernel(x)
    end
end

@time fk = randomWalkModel(1_000) # Compiling takes a while, could be optimized

@time model = SMCModel(fk)
N_particles = 2^10
nthreads = 1 #Threads.nthreads()
fullOutput = false
essThreshold = 2.0

@time smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

@time smc!(model, smcio) # Also has a long compilation time, but is somewhat independent of the particle number