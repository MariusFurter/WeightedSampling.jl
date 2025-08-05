using Random
using DataFrames
using BenchmarkTools

initialStateSampler = (rng) -> randn(rng)
randomWalkSampler = (x_in::Float64, rng) -> x_in + randn(rng)

rng = Random.default_rng()


N = 10_000
particles_df = DataFrame(weights=[0.0 for _ in 1:N])

function run_simulation_df(particles_df, rng, kernels)
    particles_df[!, :x0] .= kernels.initialStateSampler(rng)

    for i in 1:10_000
        particles_df[!, Symbol("x$i")] .= kernels.randomWalkSampler.(particles_df[!, Symbol("x0")], rng)
        #transform!(particles_df, Symbol("x0") => ByRow(x -> kernels.randomWalkSampler(x, rng)) => Symbol("x$i"))
    end
end

@btime run_simulation_df(particles_df, rng, (randomWalkSampler=randomWalkSampler, initialStateSampler=initialStateSampler)) #56ms 200k allocs

#No RNG
# 10k x 10k dynamic variables: 476ms / 150k allocs

#Explicit RNG
# 467ms / 200k allocs

# directly specifying columns solves the issue of dynamic variables!!!