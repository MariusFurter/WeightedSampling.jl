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

@time @model function randomWalkModel()
    x[0]::Float64 ~ initialState()
    for i in 1:10
        x[i]::Float64 ~ randomWalkKernel(x[i-1])
    end
end

loop_body = quote
    x[i]::Float64 ~ randomWalkKernel(x[y[i-1]])
end


body = [loop_body]

loop_steps = DrawingInferences.extract_steps(body, Main)

DrawingInferences.process_loop_statement(loop_body, :i, 5)

DrawingInferences.@sampling_to_FKStep x[i]::Float64 ~ randomWalkKernel(x[i])

fk = randomWalkModel
fk[1]
run_fk(fk)