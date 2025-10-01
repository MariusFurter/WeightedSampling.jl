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

function extract_loop_vars(loop_var)
    """Extract all symbols from a loop variable, handling both single symbols and tuple destructuring."""
    if loop_var isa Symbol
        return [loop_var]
    elseif loop_var isa Expr && loop_var.head == :tuple
        vars = Symbol[]
        for arg in loop_var.args
            if arg isa Symbol
                push!(vars, arg)
            else
                error("Only symbols are supported in tuple destructuring for loops: $arg")
            end
        end
        return vars
    else
        error("Unsupported loop variable format: $loop_var")
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
                if output_symbol in exceptions
                    return :(fill($expr, $N_sym))  # Broadcast the templated expression to all particles
                else
                    output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
                    return :($particles_sym[!, $output_expr])
                end
            else
                error("Error while parsing $expr")
            end
        elseif expr.head == :ref
            if @capture(expr, output_symbol_[index_])
                if output_symbol in exceptions
                    return :(fill($expr, $N_sym))  # Broadcast the indexed value to all particles
                else
                    return :(getindex.($particles_sym[!, $(QuoteNode(output_symbol))], $index))
                end
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
    @capture(expr, lhs_ .= rhs_)

    _, out_setter = capture_lhs(lhs)
    rhs_rewritten = replace_symbols_except(rhs, exceptions, particles_sym, N_sym)

    assign! = out_setter(particles_sym, rhs_rewritten)

    quote
        if !suppress_resampling
            resampled, ess_perc = $(WeightedSampling.resample_particles!)($particles_sym, ess_perc_min)
        end
        $assign!
        suppress_resampling = true
        current_depth += 1
        WeightedSampling.next!(progress_meter)
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
            resampled, ess_perc = $(WeightedSampling.resample_particles!)($particles_sym, ess_perc_min)
        else
            resampled = false
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
                    evidence += log(WeightedSampling.expectation(exp.(weights_scratch), $particles_sym[!, :weights]))
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
        WeightedSampling.next!(progress_meter)
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
            resampled, ess_perc = $(WeightedSampling.resample_particles!)($particles_sym, ess_perc_min)
        else
            resampled = false
        end
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            if compute_evidence
                weights_scratch .= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
                evidence += log(WeightedSampling.expectation(exp.(weights_scratch), $particles_sym[!, :weights]))
                $particles_sym[!, :weights] .+= weights_scratch
            else
                $particles_sym[!, :weights] .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
            end
            suppress_resampling = false
        end
        current_depth += 1
        WeightedSampling.next!(progress_meter)
    end
end

function rewrite_move(expr, exceptions, particles_sym, N_sym)
    @capture(expr, lhs_ << f_(args__))

    targets = extract_symbols(lhs)
    targets = setdiff(targets, exceptions)
    targets = collect(targets)

    kernel_args = map(args) do arg
        replace_symbols_except(arg, exceptions, particles_sym, N_sym)
    end

    quote
        if !suppress_resampling
            resampled, ess_perc = $(WeightedSampling.resample_particles!)($particles_sym, ess_perc_min)
        else
            resampled = false
        end
        let proposal = if hasproperty(proposals, $(QuoteNode(f)))
                proposals.$f
            else
                $f
            end
            WeightedSampling.mh!($particles_sym, proposal, $targets, current_depth, ($(kernel_args...),), smc_logpdf)
        end
        suppress_resampling = true
        current_depth += 1
        WeightedSampling.next!(progress_meter)
    end
end

function build_smc(body, exceptions, particles_sym, N_sym)
    code = quote end
    for statement in body

        if @capture(statement, lhs_ .= rhs_)
            rewritten_statement = rewrite_assignment(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ ~ f_(args__))
            rewritten_statement = rewrite_sampling(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ => f_(args__))
            rewritten_statement = rewrite_observe(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, if condition_
            body__
        end)
            e = quote
                if $condition
                    $(build_smc(body, exceptions, particles_sym, N_sym))
                end
            end
            append!(code.args, e.args)

        elseif @capture(statement, for loop_var_ in collection_
            loop_body__
        end)

            loop_vars = extract_loop_vars(loop_var)
            for var in loop_vars
                push!(exceptions, var)
            end
            e = quote
                for $loop_var in $collection
                    $(build_smc(loop_body, exceptions, particles_sym, N_sym))
                end
            end
            for var in loop_vars
                delete!(exceptions, var)
            end
            append!(code.args, e.args)

        elseif @capture(statement, lhs_ << f_(args__))
            rewritten_statement = rewrite_move(statement, exceptions, particles_sym, N_sym)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ = rhs_)
            lhs_vars = extract_symbols(lhs)
            push!(exceptions, lhs_vars...)
            push!(code.args, statement)

        else
            push!(code.args, statement)
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

    arg_exceptions = Set{Symbol}((args..., kwarg_names...))
    reserved_names = Set{Symbol}([:undef])
    exceptions = union(arg_exceptions, reserved_names)

    name! = Symbol(name, "!")

    N_sym = :N
    particles_sym = :particles

    return esc(quote
        function $name!($(args...); $(kwargs...),
            $particles_sym,
            kernels=nothing,
            proposals=nothing,
            ess_perc_min=0.5::Float64,
            compute_evidence=true::Bool,
            show_progress=true::Bool)

            if kernels === nothing
                kernels = WeightedSampling.default_kernels
            else
                kernels = merge(WeightedSampling.default_kernels, kernels)
            end

            if proposals === nothing
                proposals = WeightedSampling.default_proposals
            else
                proposals = merge(WeightedSampling.default_proposals, proposals)
            end

            $N_sym = WeightedSampling.nrow($particles_sym)

            progress_meter = WeightedSampling.ProgressUnknown(desc="Steps performed:", dt=0.1, showspeed=true, color=:blue, enabled=show_progress)

            $(build_logpdf(body, exceptions, N_sym))

            suppress_resampling = true
            resampled = false
            ess_perc = 1.0
            weights_scratch = zeros($N_sym)
            current_depth = 0
            evidence = 0.0

            $(build_smc(body, exceptions, particles_sym, N_sym))
            WeightedSampling.finish!(progress_meter)

            return evidence
        end

        function $name($(args...); $(kwargs...),
            n_particles=1_000::Int64,
            kernels=nothing,
            proposals=nothing,
            ess_perc_min=0.5::Float64,
            compute_evidence=true::Bool,
            show_progress=true::Bool)

            $particles_sym = WeightedSampling.DataFrame(weights=zeros(n_particles))

            evidence = $name!($(args...); $(kwargs...),
                $particles_sym=$particles_sym,
                kernels=kernels,
                proposals=proposals,
                ess_perc_min=ess_perc_min,
                compute_evidence=compute_evidence,
                show_progress=show_progress)

            return $particles_sym, evidence
        end
    end)
end