using Distributions
using BenchmarkTools
using Random
using MacroTools


initialStateSampler = (rng) -> randn(rng)
randomWalkSampler = (x_in::Float64, rng) -> x_in + randn(rng)

rng = Random.default_rng()

struct ParticleList{P}
    samples::Vector{P}
    weights::Vector{Float64}
end

@generated function update!(p::ParticleList{P}, kernel::Val{K}, args, output::Val{O}, rng) where {P,O,K}
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

variables = Dict(Symbol("x$i") => Float64 for i in 0:1000)

function make_struct(name, variables)
    fields = [:($field::$type) for (field, type) in variables]
    struct_code = quote
        mutable struct $name
            $(fields...)
        end
    end
    Core.eval(Main, struct_code)
end

make_struct(:TimeParticle, variables)

p = TimeParticle((0.0 for i in 0:1000)...)

particles = ParticleList{TimeParticle}([p for _ in 1:1000], [1.0 for _ in 1:1000])

function run_simulation(particles, rng, kernels)
    update!(particles, Val(kernels.initialStateSampler), (), Val(Symbol("x0")), rng)

    for i in 1:1000
        update!(particles, Val(kernels.randomWalkSampler), (Symbol("x$(i-1)"),), Val(Symbol("x$i")), rng)
    end
end

@time run_simulation(particles, rng, (randomWalkSampler=randomWalkSampler, initialStateSampler=initialStateSampler)) # 1000p x 1000t: 13k allocs (independent of particle number, seem entirely due to creating of Symbols and Vals at runtime), 5s compile, 180ms.

