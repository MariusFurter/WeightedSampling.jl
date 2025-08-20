using Revise
using DrawingInferences
using MacroTools
using DataFrames
using Distributions
using StatsBase

#Write mh! properly + assemble in rewrites.jl

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

    return DrawingInferences.build_logpdf_difference(body, exceptions, :N)
end

ldiffs = @logpdf_diff function bla()
    for i in 1:10
        x ~ Normal(i, 1)
    end
end

current_particles = DataFrame(x=fill(0.0, 10), y=fill(0.0, 10), z=fill(0.0, 10))
proposed_particles = DataFrame(x=fill(1.0, 10), y=fill(0.0, 10), z=fill(0.0, 10))

ldiffs(current_particles, proposed_particles, [:x], 0, DrawingInferences.default_kernels)