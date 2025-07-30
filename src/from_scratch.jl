using Distributions
using BenchmarkTools
using Random
using MacroTools


initialStateSampler = (rng) -> randn(rng)
randomWalkSampler = (x_in::Float64, rng) -> x_in + randn(rng)

rng = Random.default_rng()

mutable struct Particle
    x::Float64
end

struct ParticleList
    samples::Vector{Particle}
    weights::Vector{Float64}
end

### Generated functions!
@generated function randomWalk!(p::Particle, args, output::Val{O}, rng) where {O}
    # Get the first element type
    args_expr = []
    for (i, type) in enumerate(args.parameters)
        if type == Symbol
            push!(args_expr, :(getproperty(p, args[$i])))
        else
            push!(args_expr, :(args[$i]))
        end
    end

    @show args_expr

    quote
        val = $(args_expr[1]) + randn(rng)
        setproperty!(p, O, val)
    end
end

@generated function update!(p::ParticleList, kernel::Val{K}, args, output::Val{O}, rng) where {O,K}
    # Get the first element type
    args_expr = []
    for (i, type) in enumerate(args.parameters)
        if type == Symbol
            push!(args_expr, :(getproperty(particle, args[$i])))
        else
            push!(args_expr, :(args[$i]))
        end
    end

    quote
        for particle in p.samples
            val = $K($(args_expr...), rng)
            setproperty!(particle, O, val)
        end
    end
end

p = Particle(0.0)

particles = ParticleList([p for _ in 1:1000], [1.0 for _ in 1:1000])

function run_simulation(particles, rng, kernels)

    update!(particles, Val(kernels.initialStateSampler), (), Val(:x), rng)
    for i in 1:100
        update!(particles, Val(kernels.randomWalkSampler), (:x,), Val(:x), rng)
    end
end

@btime run_simulation(particles, rng, (randomWalkSampler=randomWalkSampler, initialStateSampler=initialStateSampler)) # 1000p x 100t: No allocs 430us.