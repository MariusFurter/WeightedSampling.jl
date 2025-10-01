using Revise
using WeightedSampling
using MacroTools
using DataFrames
using Distributions
using StatsBase
using CairoMakie

using LinearAlgebra

UniformNormal = SMCKernel(
    () -> rand(),
    (x) -> logpdf(Beta(2, 10), x),
    (x) -> logpdf(Beta(2, 10), x),
)

@smc function bla(ess_list)
    x ~ UniformNormal()
    y .= 3.0
    z = 0.1
    w ~ Normal(z, 0.1)
    for i in 1:1_000
        0.2 => Normal(y, z)
        if resampled
            x << autoRW()
        end
        push!(ess_list, ess_perc)
    end
end

ess_list = Float64[]

samples, evidence = @time bla(ess_list, n_particles=1_000, ess_perc_min=0.5)

ess_list

hist(samples.x, normalization=:pdf, bins=100, weights=exp.(samples.weights))
hist!(samples.y, normalization=:pdf, bins=100, weights=exp.(samples.weights))
xs = range(start=0, stop=1, length=100)
lines!(xs, pdf.(Beta(2, 10), xs), color=:orange, linewidth=2)
current_figure()

code |> MacroTools.prettify

macro logpdf_diff(expr)
    if @capture(expr, function name_(args__; kwargs__)
        body__
    end)
    elseif @capture(expr, function name_(; kwargs__)
        body__
    end)
        args = Symbol[]
    elseif @capture(expr, function name_(args__)
        body__
    end)
        kwargs = Expr[]
    else
        error("Expression must be a function definition")
    end

    kwarg_names = WeightedSampling.extract_kwarg_names(kwargs)

    arg_exceptions = Set{Symbol}((args..., kwarg_names...))
    reserved_names = Set{Symbol}([:undef])
    exceptions = union(arg_exceptions, reserved_names)

    return WeightedSampling.build_logpdf(body, exceptions, :N)
end

smc_logpdf = @logpdf_diff function bla()
    for i in 1:10
        x ~ Normal(i, 1)
    end
end


particles = DataFrame(x=fill(0.0, 10), y=fill(0.0, 10), z=fill(0.0, 10))