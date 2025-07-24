using DrawingInferences
using Distributions
using BenchmarkTools

function makeInitialState()
    output_types = [Float64]
    kernel = () -> Normal(0.0, 1.0)
    sampler = (rng) -> rand(rng, kernel())
    lpdf = () -> x -> logpdf(kernel(), x)
    weighter = x -> 0.0
    return WeightedSampler(output_types, sampler, lpdf, weighter)
end

initialState = makeInitialState()

function makeRandomWalkKernel()
    output_types = [Float64]
    kernel = x -> Normal(x, 1.0)
    sampler = (x, rng) -> rand(rng, kernel(x))
    lpdf = xin -> xout -> logpdf(kernel(xin), xout)
    weighter = x -> 0.0
    return WeightedSampler(output_types, sampler, lpdf, weighter)
end

randomWalkKernel = makeRandomWalkKernel()


@fk function randomWalkModel(T::Int64)
    x = initialState()
    for i in 1:T
        x = randomWalkKernel(x)
    end
end


@time fk = randomWalkModel(1000)
@time model = makeSMCModel(fk)

N_particles = 2^10
nthreads = 1# Threads.nthreads()
fullOutput = false
essThreshold = 2.0

smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

@time smc!(model, smcio)