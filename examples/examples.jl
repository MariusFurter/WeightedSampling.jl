using DrawingInferences
using Distributions
using BenchmarkTools
using Random
using MacroTools

### Hand code steps

initialState = WeightedSampler(
    (rng) -> randn(rng),
    (x::Float64) -> 0.0,
    () -> Float64,
)

initialStep = @fkstep x = initialState()

# Create instance with WeightedSampler
randomWalkKernel = WeightedSampler(
    (x::Float64, rng) -> rand(rng, Normal(x, 1)),
    (x::Float64) -> 0.0,
    () -> Float64,
)

randomStep = @fkstep x = randomWalkKernel(x)

fk = FKModel((initialStep, (randomStep for _ in 1:10)...))

model = SMCModel(fk)

N_particles = 2^10
nthreads = 1# Threads.nthreads()
fullOutput = false
essThreshold = 2.0

smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

@time smc!(model, smcio)


### Full code

@fk function randomWalkModel(T::Int64)
    x = initialState()
    for i in 1:T
        x = randomWalkKernel(x)
    end
end


@time fk = randomWalkModel(100)
@time model = SMCModel(fk)

N_particles = 2^10
nthreads = 1# Threads.nthreads()
fullOutput = false
essThreshold = 2.0

smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

@time smc!(model, smcio)