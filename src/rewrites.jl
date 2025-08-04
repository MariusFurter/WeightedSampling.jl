using MacroTools
using Distributions
using Random

function replace_symbols(expr, particle_sym, exceptions)
    if expr isa Symbol && !(expr in exceptions)
        return :(getproperty($particle_sym, $(QuoteNode(expr))))
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, expr.args[1], map(x -> replace_symbols(x, particle_sym, exceptions), expr.args[2:end])...)
        else

            return Expr(expr.head, map(x -> replace_symbols(x, particle_sym, exceptions), expr.args)...)
        end
    else
        return expr
    end
end

function capture_lhs(expr)

    if expr isa Symbol
        output_expr = QuoteNode(expr)
    elseif @capture(expr, output_symbol_{index_})
        output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
    else
        error("Left-hand side must be a variable name `x` or an indexed variable `x{i}`")
    end

    #ToDo: Also capture type annotations
    output_type = Any

    return output_expr, output_type
end

function rewrite_assignment(expr, exceptions)
    @capture(expr, lhs_ = rhs_)

    output_expr, _ = capture_lhs(lhs)

    particle_sym = gensym("particle")
    rhs_rewritten = replace_symbols(rhs, particle_sym, exceptions)

    quote
        for $particle_sym in particles.samples
            val = $(rhs_rewritten)
            setproperty!($particle_sym, $output_expr, val)
        end

    end
end

function rewrite_sampling(expr, exceptions)
    @capture(expr, lhs_ ~ f_(args__))

    output_expr, _ = capture_lhs(lhs)

    particle_sym = gensym("particle")
    args_rewritten = map(args) do arg
        replace_symbols(arg, particle_sym, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            for (i, $particle_sym) in enumerate(particles.samples)
                val = kernel.sampler($(args_rewritten...), rng)
                setproperty!($particle_sym, $output_expr, val)
                if kernel.weighter !== nothing
                    particles.weights[i] += kernel.weighter($(args_rewritten...), val)
                end
            end
        end
    end
end

function rewrite_observe(expr, exceptions)
    @capture(expr, lhs_ -> f_(args__))

    particle_sym = gensym("particle")
    lhs_rewritten = replace_symbols(lhs, particle_sym, exceptions)
    args_rewritten = map(args) do arg
        replace_symbols(arg, particle_sym, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            for (i, $particle_sym) in enumerate(particles.samples)
                val = $lhs_rewritten
                particles.weights[i] += kernel.logpdf($(args_rewritten...), val)
            end
        end
    end
end

function build_smc(body, exceptions)
    code = quote end
    for statement in body

        if @capture(statement, lhs_ = rhs_)
            rewritten_statement = rewrite_assignment(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ ~ f_(args__))
            rewritten_statement = rewrite_sampling(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ -> f_(args__))
            rewritten_statement = rewrite_observe(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            push!(exceptions, loop_var)
            e = quote
                for $loop_var in $start:$stop
                    $(build_smc(loop_body, exceptions))
                end
            end
            delete!(exceptions, loop_var)
            append!(code.args, e.args)
        end
    end
    return code
end

function extract_variables(body)
    code = quote
        variables = Dict{Symbol,DataType}()
    end
    for statement in body

        if @capture(statement, lhs_ = rhs_) || @capture(statement, lhs_ ~ f_(args__))

            output_expr, output_type = capture_lhs(lhs)
            e = quote
                variables[$output_expr] = $output_type
            end
            append!(code.args, e.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            e = quote
                for $loop_var in $start:$stop
                    $(extract_variables(loop_body))
                end
            end
            append!(code.args, e.args)
        end
    end
    return code
end

macro parse(expr)
    @capture(expr, function name_()
        body__
    end) || error("Expression must be a function definition")

    exceptions = Set{Symbol}()

    # Later: replace with default kernels provided by DrawingInferences
    return esc(quote
        function $name(particles, kernels=nothing, rng=Random.default_rng())
            $(build_smc(body, exceptions))
        end
    end)
end

struct SMCParticles{P}
    samples::Vector{P}
    weights::Vector{Float64}
end

struct SMCKernel{S,L,W}
    sampler::S
    logpdf::L
    weighter::W
end

initialKernel = SMCKernel(
    (rng) -> randn(rng),
    (x) -> logpdf(Normal(0, 1), x),
    nothing
)

walkKernel = SMCKernel(
    (x_in::Float64, rng) -> x_in + randn(rng),
    (x_in::Float64, x_out::Float64) -> logpdf(Normal(x_in, 1), x_out),
    (x_in::Float64, x_out::Float64) -> 0.0
)

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

mutable struct XYParticle
    x::Float64
    y1::Float64
    y2::Float64
end

my_particles = SMCParticles{XYParticle}([XYParticle(0.0, 0.0, 0.0) for _ in 1:1000], [0.0 for _ in 1:1000])

my_rng = Random.default_rng()

@parse function bla!()
    x ~ initialKernel()
    for i in 1:100
        x ~ walkKernel(x)
    end
    5.0 -> walkKernel(x)
end

@time bla!(my_particles, my_kernels, my_rng)

## Attempt to get variables dynamically. Probably best to parse code in a second way that gets variables and types since this can be executed before runtime.

##Pro dummy run: You also get the types for free.
