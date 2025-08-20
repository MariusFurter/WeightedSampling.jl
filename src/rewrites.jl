# Todos:
# - Think about variable scopes. Maybe replace symbols in constructed function body with gensyms, or use let blocks.

# - Support more general for loops.

# - Recursion & composition with other smc functions more generally. Wrap smc function in type and have a case distinction in code.
# -> Check that @smc generated function works with vector arguments
# -> Add a way to specify return values
# -> How does evidence compose?
# -> What about moves?
# -> then use mutating function on current particles / could also use a non-mutating version if we don't want to keep track of the intermediate values. Probably should be black-boxed.
# -> Find a way to introduce namespacing in a controlled way (might be unnecessary with black-boxing)

function extract_symbols(expr)
    if expr isa Symbol
        return Set{Symbol}([expr])
    elseif expr isa Expr
        if expr.head == :call
            return union(extract_symbols.(expr.args[2:end])...)
        else
            return union(extract_symbols.(expr.args)...)
        end
    else
        return Set{Symbol}()
    end
end

function replace_symbols_except(expr, exceptions, particles_sym, N_sym)
    if expr isa Symbol && !(expr in exceptions)
        return :($particles_sym[!, $(QuoteNode(expr))])
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, :broadcast, expr.args[1], map(x -> replace_symbols_except(x, exceptions, particles_sym, N_sym), expr.args[2:end])...)
        elseif expr.head == :tuple
            return :(fill($expr, $N_sym))
        elseif expr.head == :vect
            symbols = collect(extract_symbols(expr))
            if all(map(s -> s in exceptions, symbols))
                return :(fill($expr, $N_sym))
            else
                return Expr(:call, :broadcast, :MVector, map(x -> replace_symbols_except(x, exceptions, particles_sym, N_sym), expr.args)...) #MVector avoids some overhead of vcat.
            end
        elseif expr.head == :curly
            if @capture(expr, output_symbol_{index_})
                output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                return :($particles_sym[!, $output_expr])
            else
                error("Error while parsing $expr")
            end
        elseif expr.head == :ref
            if @capture(expr, output_symbol_[index_])
                return :(getindex.($particles_sym[!, $(QuoteNode(output_symbol))], $index))
            else
                error("Error while parsing $expr")
            end
        else
            return Expr(expr.head, map(x -> replace_symbols_except(x, exceptions, particles_sym, N_sym), expr.args)...)
        end
    else
        return :(fill($expr, $N_sym))
    end
end

function replace_symbols_in(expr, to_replace, particles_sym, N_sym)
    if expr isa Symbol && expr in to_replace
        return :($particles_sym[!, $(QuoteNode(expr))])
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, :broadcast, expr.args[1], map(x -> replace_symbols_in(x, to_replace, particles_sym, N_sym), expr.args[2:end])...)
        elseif expr.head == :tuple
            return :(fill($expr, $N_sym))
        elseif expr.head == :vect
            symbols = collect(extract_symbols(expr))
            if all(map(s -> s in exceptions, symbols))
                return :(fill($expr, $N_sym))
            else
                return Expr(:call, :broadcast, :MVector, map(x -> replace_symbols_except(x, exceptions, particles_sym, N_sym), expr.args)...) #MVector avoids some overhead of vcat.
            end
        elseif expr.head == :curly
            if @capture(expr, output_symbol_{index_})
                output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                return :($particles_sym[!, $output_expr])
            else
                error("Error while parsing $expr")
            end
        elseif expr.head == :ref
            if @capture(expr, output_symbol_[index_])
                return :(getindex.($particles_sym[!, $(QuoteNode(output_symbol))], $index))
            else
                error("Error while parsing $expr")
            end
        else
            return Expr(expr.head, map(x -> replace_symbols_in(x, to_replace, particles_sym, N_sym), expr.args)...)
        end
    else
        return :(fill($expr, $N_sym))
    end
end

function capture_lhs(expr)
    if expr isa Symbol
        output_expr = QuoteNode(expr)
        getter = (particles_sym) -> :($particles_sym[!, $output_expr])
        setter = (particles_sym, values_expr) -> :($particles_sym[!, $output_expr] .= $values_expr)

    elseif @capture(expr, output_symbol_{index_})
        output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
        getter = (particles_sym) -> :($particles_sym[!, $output_expr])
        setter = (particles_sym, values_expr) -> :($particles_sym[!, $output_expr] .= $values_expr)

    elseif @capture(expr, output_symbol_[index_])
        output_expr = QuoteNode(output_symbol)
        getter = (particles_sym) -> :(getindex.($particles_sym[!, $output_expr], $index))
        setter = (particles_sym, values_expr) -> :(setindex!.($particles_sym[!, $output_expr], $values_expr, $index))
    else
        error("Left-hand side must be a variable name `x` or an indexed variable `x{i}` or `x[i]`.")
    end

    return getter, setter
end

function rewrite_assignment(expr, exceptions, particles_sym, N_sym)
    @capture(expr, lhs_ = rhs_)

    _, out_setter = capture_lhs(lhs)
    rhs_rewritten = replace_symbols_except(rhs, exceptions, particles_sym, N_sym)

    assign! = out_setter(particles_sym, rhs_rewritten)

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)($particles_sym, ess_perc_min)
        end
        $assign!
        suppress_resampling = true
        current_depth += 1
        DrawingInferences.next!(progress_meter)
    end
end

function rewrite_sampling(expr, exceptions, particles_sym, N_sym)
    @capture(expr, lhs_ ~ f_(args__))

    out_getter, out_setter = capture_lhs(lhs)
    args_rewritten = map(args) do arg
        replace_symbols_except(arg, exceptions, particles_sym, N_sym)
    end

    sample! = out_setter(particles_sym, :(kernel.sampler.($(args_rewritten...))))
    output_value = out_getter(particles_sym)

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)($particles_sym, ess_perc_min)
        end
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            $sample!
            if kernel.weighter !== nothing
                if compute_evidence
                    weights_scratch = kernel.weighter.($(args_rewritten...), $output_value)
                    evidence += log(DrawingInferences.expectation(exp.(weights_scratch), $particles_sym[!, :weights]))
                    $particles_sym[!, :weights] .+= weights_scratch
                else
                    $particles_sym[!, :weights] .+= kernel.weighter.($(args_rewritten...), $output_value)
                end
                suppress_resampling = false
            else
                suppress_resampling = true
            end
        end
        current_depth += 1
        DrawingInferences.next!(progress_meter)
    end
end

function rewrite_observe(expr, exceptions, particles_sym, N_sym)
    @capture(expr, lhs_ => f_(args__))

    lhs_rewritten = replace_symbols_except(lhs, exceptions, particles_sym, N_sym)
    args_rewritten = map(args) do arg
        replace_symbols_except(arg, exceptions, particles_sym, N_sym)
    end

    quote
        if !suppress_resampling
            $(DrawingInferences.resample_particles!)($particles_sym, ess_perc_min)
        end
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            if compute_evidence
                weights_scratch .= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
                evidence += log(DrawingInferences.expectation(exp.(weights_scratch), $particles_sym[!, :weights]))
                $particles_sym[!, :weights] .+= weights_scratch
            else
                $particles_sym[!, :weights] .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
            end
            suppress_resampling = false
        end
        current_depth += 1
        DrawingInferences.next!(progress_meter)
    end
end

function rewrite_move(expr, exceptions, particles_sym, N_sym)
    @capture(expr, lhs_ << f_(args__))

    targets = extract_symbols(lhs)
    targets = setdiff(targets, exceptions)

    kernel_args = map(args) do arg
        replace_symbols_except(arg, exceptions, particles_sym, N_sym)
    end

    quote
        let proposal = if hasproperty(proposals, $(QuoteNode(f)))
                proposals.$f
            else
                $f
            end
            DrawingInferences.mh!($particles_sym, proposal, $targets, current_depth, ($(kernel_args...),), smc_logpdf)
        end
        current_depth += 1
        DrawingInferences.next!(progress_meter)
    end
end

function build_smc(body, exceptions, particles_sym, N_sym)
    code = quote end
    for statement in body

        if @capture(statement, lhs_ = rhs_)
            rewritten_statement = rewrite_assignment(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ ~ f_(args__))
            rewritten_statement = rewrite_sampling(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ => f_(args__))
            rewritten_statement = rewrite_observe(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            push!(exceptions, loop_var)
            e = quote
                for $loop_var in $start:$stop
                    $(build_smc(loop_body, exceptions, particles_sym, N_sym))
                end
            end
            delete!(exceptions, loop_var)
            append!(code.args, e.args)

        elseif @capture(statement, lhs_ << f_(args__))
            rewritten_statement = rewrite_move(statement, exceptions, particles_sym, N_sym)

            append!(code.args, rewritten_statement.args)
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

function build_step_counter(body, exceptions, steps_sym)
    code = quote end
    for statement in body
        if @capture(statement, lhs_ = rhs_) || @capture(statement, lhs_ ~ f_(args__)) || @capture(statement, lhs_ => f_(args__)) || @capture(statement, lhs_ << f_(args__))
            e = quote
                $steps_sym += 1
            end
            append!(code.args, e.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            push!(exceptions, loop_var)
            e = quote
                for $loop_var in $start:$stop
                    $(build_step_counter(loop_body, exceptions, steps_sym))
                end
            end
            delete!(exceptions, loop_var)
            append!(code.args, e.args)

        else
            error("Unsupported statement type: $statement")
        end

    end
    code
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

    arg_exceptions = Set{Symbol}((args..., kwarg_names...))
    reserved_names = Set{Symbol}([:undef])
    exceptions = union(arg_exceptions, reserved_names)

    name! = Symbol(name, "!")

    N_sym = :N
    particles_sym = :particles

    return esc(quote
        function $name!($(args...); $(kwargs...), $particles_sym, kernels=nothing, proposals=nothing, ess_perc_min=0.5::Float64, compute_evidence=true::Bool)

            if kernels === nothing
                kernels = DrawingInferences.default_kernels
            else
                kernels = merge(DrawingInferences.default_kernels, kernels)
            end

            if proposals === nothing
                proposals = DrawingInferences.default_proposals
            else
                proposals = merge(DrawingInferences.default_proposals, proposals)
            end

            $N_sym = DrawingInferences.nrow($particles_sym)

            steps = 0
            $(build_step_counter(body, exceptions, :steps))
            progress_meter = DrawingInferences.Progress(steps; dt=0.1,
                barglyphs=DrawingInferences.BarGlyphs("[=>.]"),
                barlen=40,
                color=:blue)

            $(build_logpdf(body, exceptions, N_sym))

            suppress_resampling = true
            weights_scratch = zeros($N_sym)
            current_depth = 0
            evidence = 0.0

            $(build_smc(body, exceptions, particles_sym, N_sym))

            return evidence
        end

        function $name($(args...); $(kwargs...), n_particles=1_000::Int64, kernels=nothing, proposals=nothing, ess_perc_min=0.5::Float64, compute_evidence=true::Bool)

            $particles_sym = DrawingInferences.DataFrame(weights=zeros(n_particles))

            evidence = $name!($(args...); $(kwargs...),
                $particles_sym=$particles_sym,
                kernels=kernels,
                proposals=proposals,
                ess_perc_min=ess_perc_min,
                compute_evidence=compute_evidence)

            return $particles_sym, evidence
        end
    end)
end