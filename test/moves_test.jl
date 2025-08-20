using Revise
using DrawingInferences
using MacroTools
using DataFrames
using Distributions
using StatsBase
using CairoMakie

UniformNormal = SMCKernel(
    () -> rand(),
    (x) -> logpdf(Exponential(1), x),
    nothing,
)

@smc function bla(T)
    x ~ UniformNormal()
    for i in 1:T
        x << RW()
    end
end

T = 10
samples, evidence = @time bla(T)

hist(samples.x, normalization=:pdf)
xs = range(-3, stop=3, length=100)
lines!(xs, pdf(Exponential(1), xs))
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