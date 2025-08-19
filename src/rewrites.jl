# Todos:
# - Replace symbols in constructed function body with gensyms 

# - Evidence -> add variable to function body and incrementally update it at resampling steps.

# - Get vector-valued variables working.

# - Recursion & composition with other smc functions more generally. Wrap smc function in type and have a case distinction in code.
# -> Check that @smc generated function works with vector arguments
# -> Add a way to specify return values
# -> How does evidence compose?
# -> What about moves?
# -> then use mutating function on current particles / could also use a non-mutating version if we don't want to keep track of the intermediate values. Probably should be black-boxed.
# -> Find a way to introduce namespacing in a controlled way (might be unnecessary with black-boxing)

function replace_symbols_except(expr, exceptions)
    if expr isa Symbol && !(expr in exceptions)
        return :(particles[!, $(QuoteNode(expr))])
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, :broadcast, expr.args[1], map(x -> replace_symbols_except(x, exceptions), expr.args[2:end])...)
        elseif expr.head == :curly
            if @capture(expr, output_symbol_{index_})
                output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                return :(particles[!, $output_expr])
            else
                error("Only single expression allowed in curly braces")
            end
        else
            return Expr(expr.head, map(x -> replace_symbols_except(x, exceptions), expr.args)...)
        end
    else
        return expr
    end
end

function replace_symbols_in(expr, to_replace)
    if expr isa Symbol && expr in to_replace
        return :(particles[!, $(QuoteNode(expr))])
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, :broadcast, expr.args[1], map(x -> replace_symbols_in(x, to_replace), expr.args[2:end])...)
        elseif expr.head == :curly
            if @capture(expr, output_symbol_{index_})
                output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                return :(particles[!, $output_expr])
            else
                error("Only single expression allowed in curly braces")
            end
        else
            return Expr(expr.head, map(x -> replace_symbols_in(x, to_replace), expr.args)...)
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

    return output_expr
end

function rewrite_assignment(expr, exceptions)
    @capture(expr, lhs_ = rhs_)

    output_expr = capture_lhs(lhs)
    rhs_rewritten = replace_symbols_except(rhs, exceptions)

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)(particles, ess_perc_min)
        end
        particles[!, $output_expr] .= $(rhs_rewritten)
        suppress_resampling = true
    end
end

function rewrite_sampling(expr, exceptions)
    @capture(expr, lhs_ ~ f_(args__))

    output_expr = capture_lhs(lhs)
    args_rewritten = map(args) do arg
        replace_symbols_except(arg, exceptions)
    end

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)(particles, ess_perc_min)
        end
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            particles[!, $output_expr] .= kernel.sampler.($(args_rewritten...))
            if kernel.weighter !== nothing
                if compute_evidence
                    weights_scratch = kernel.weighter.($(args_rewritten...), particles[!, $output_expr])
                    evidence += log(DrawingInferences.expectation(exp.(weights_scratch), particles[!, :weights]))
                    particles[!, :weights] .+= weights_scratch
                else
                    particles[!, :weights] .+= kernel.weighter.($(args_rewritten...), particles[!, $output_expr])
                end
                suppress_resampling = false
            else
                suppress_resampling = true
            end
        end
    end
end

function rewrite_observe(expr, exceptions)
    @capture(expr, lhs_ => f_(args__))

    lhs_rewritten = replace_symbols_except(lhs, exceptions)
    args_rewritten = map(args) do arg
        replace_symbols_except(arg, exceptions)
    end

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)(particles, ess_perc_min)
        end
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            if compute_evidence
                weights_scratch .= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
                evidence += log(DrawingInferences.expectation(exp.(weights_scratch), particles[!, :weights]))
                particles[!, :weights] .+= weights_scratch
            else
                particles[!, :weights] .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
            end
            suppress_resampling = false
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

        elseif @capture(statement, lhs_ => f_(args__))
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

        else
            error("Unsupported statement type: $statement")
        end
    end
    return code
end

function extract_kwarg_names(kwargs)
    kwarg_names = Symbol[]
    for kwarg in kwargs
        if @capture(kwarg, name_ = value_)
            push!(kwarg_names, name)
        elseif @capture(kwarg, name_Symbol)
            push!(kwarg_names, name)
        else
            error("Unsupported keyword argument: $kwarg")
        end
    end
    return kwarg_names
end

macro smc(expr)
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

    kwarg_names = extract_kwarg_names(kwargs)

    exceptions = Set{Symbol}((args..., kwarg_names...))

    name! = Symbol(name, "!")

    return esc(quote
        function $name!($(args...); $(kwargs...), particles, kernels=nothing, ess_perc_min=0.5::Float64, compute_evidence=true::Bool)

            if kernels === nothing
                kernels = DrawingInferences.default_kernels
            else
                kernels = merge(DrawingInferences.default_kernels, kernels)
            end

            suppress_resampling = true
            weights_scratch = zeros(DrawingInferences.nrow(particles))
            evidence = 0.0

            $(build_smc(body, exceptions))
            return evidence
        end

        function $name($(args...); $(kwargs...), n_particles=1_000::Int64, kernels=nothing, ess_perc_min=0.5::Float64, compute_evidence=true::Bool)

            if kernels === nothing
                kernels = DrawingInferences.default_kernels
            else
                kernels = merge(DrawingInferences.default_kernels, kernels)
            end

            suppress_resampling = true
            weights_scratch = zeros(n_particles)
            evidence = 0.0

            particles = DrawingInferences.DataFrame(weights=zeros(n_particles))
            $(build_smc(body, exceptions))
            return particles, evidence
        end
    end)
end