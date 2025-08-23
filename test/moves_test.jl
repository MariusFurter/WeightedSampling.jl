using Revise
using DrawingInferences
using MacroTools
using DataFrames
using Distributions
using StatsBase
using CairoMakie

UniformNormal = SMCKernel(
    () -> rand(),
    (x) -> logpdf(Beta(2, 10), x),
    (x) -> logpdf(Beta(2, 10), x),
)

@smc function bla()
    x ~ Normal(0, 1)
    y ~ Normal(0, 1)
    for i in 1:1_000
        0.2 => Normal(x, 0.1)
        0.3 => Normal(y, 0.2)
        if diversity(particles, [:x, :y]) < 0.5
            (x, y) << autoRW()
        end
    end
end

samples, evidence = @time bla(n_particles=1_000, ess_perc_min=0.5)

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

    kwarg_names = DrawingInferences.extract_kwarg_names(kwargs)

    arg_exceptions = Set{Symbol}((args..., kwarg_names...))
    reserved_names = Set{Symbol}([:undef])
    exceptions = union(arg_exceptions, reserved_names)

    return DrawingInferences.build_logpdf(body, exceptions, :N)
end

smc_logpdf = @logpdf_diff function bla()
    for i in 1:10
        x ~ Normal(i, 1)
    end
end


particles = DataFrame(x=fill(0.0, 10), y=fill(0.0, 10), z=fill(0.0, 10))