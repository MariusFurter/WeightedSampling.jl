using MacroTools

function replace_symbols(expr, exceptions)
    if expr isa Symbol && !(expr in exceptions)
        return :(particles[!, $(QuoteNode(expr))])
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, :broadcast, expr.args[1], map(x -> replace_symbols(x, exceptions), expr.args[2:end])...)
        elseif expr.head == :curly
            if @capture(expr, output_symbol_{index_})
                output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                return :(particles[!, $output_expr])
            else
                error("Only single expression allowed in curly braces")
            end

        else
            return Expr(expr.head, map(x -> replace_symbols(x, exceptions), expr.args)...)
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
    rhs_rewritten = replace_symbols(rhs, exceptions)

    quote
        particles[!, $output_expr] .= $(rhs_rewritten)
    end
end

function rewrite_sampling(expr, exceptions)
    @capture(expr, lhs_ ~ f_(args__))

    output_expr = capture_lhs(lhs)
    args_rewritten = map(args) do arg
        replace_symbols(arg, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            particles[!, $output_expr] .= kernel.sampler.($(args_rewritten...))
            if kernel.weighter !== nothing
                particles[!, :weights] .+= kernel.weighter.($(args_rewritten...), particles[!, $output_expr])
                $(DrawingInferences.resample_particles!)(particles, ess_perc_min)
            end
        end
    end
end

function rewrite_observe(expr, exceptions)
    @capture(expr, lhs_ -> f_(args__))

    lhs_rewritten = replace_symbols(lhs, exceptions)
    args_rewritten = map(args) do arg
        replace_symbols(arg, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            particles[!, :weights] .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
            $(DrawingInferences.resample_particles!)(particles, ess_perc_min)
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

        else
            error("Unsupported statement type: $statement")
        end
    end
    return code
end

macro smc(expr)
    @capture(expr, function name_()
        body__
    end) || error("Expression must be a function definition")

    exceptions = Set{Symbol}()

    name! = Symbol(name, "!")

    # Later: replace with default kernels provided by DrawingInferences
    return esc(quote
        function $name!(particles, kernels=nothing, ess_perc_min=0.5::Float64)
            $(build_smc(body, exceptions))
            return nothing
        end
    end)
end