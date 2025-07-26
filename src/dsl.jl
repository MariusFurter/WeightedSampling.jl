using MacroTools

function get_variables(expr)
    vars = Set{Symbol}()

    function recursive_walk(e)
        if e isa Symbol
            push!(vars, e)
        elseif e isa Expr
            if e.head === :call
                for arg in e.args[2:end]  # Skip the function name
                    recursive_walk(arg)
                end
            else
                for arg in e.args
                    recursive_walk(arg)
                end
            end
        end
    end

    recursive_walk(expr)
    return collect(vars)
end

function replace_refs(expr)
    MacroTools.postwalk(expr) do e
        if e isa Expr && e.head === :ref  # Handle indexed variables like x[i] or x{i}
            return Symbol("$(e.args[1])[$(e.args[2])]")
        elseif e isa Expr && e.head === :curly  # Handle variables with curly braces like x{i}
            return Symbol("$(e.args[1]){$(e.args[2])}")
        else
            return e
        end
    end
end

function variables_to_particle(vars, expr)
    MacroTools.postwalk(expr) do x
        x in vars ? :(particle[$(QuoteNode(x))]) : x
    end
end

function capture_lhs(expr)

    function validate_output(output)
        output isa Symbol ||
            @capture(output, output_symbol_[index_]) ||
            @capture(output, output_symbol_{index_}) ||
            error("Left-hand side must be a variable name `x` or an indexed variable `x[i]` or `x{i}`")
    end

    if @capture(expr, output_::output_type_)
        if validate_output(output)
            output = Symbol(output)
        end
        output_type = eval(output_type)
        @assert output_type isa DataType "Output type must be a DataType"

    elseif validate_output(expr)
        output = Symbol(expr)
        output_type = Any
    end

    return output, output_type
end

function process_args(args)
    inputs = []

    args = map(args) do arg
        arg = replace_refs(arg)
        arg_inputs = get_variables(arg)
        push!(inputs, arg_inputs)
        variables_to_particle(arg_inputs, arg)
    end

    inputs = reduce(union, inputs, init=Symbol[])

    return inputs, args
end

function assignment_to_FKStep(expr)
    @capture(expr, lhs_ = rhs_) || error("Expression must be an assignment `x = expr`")

    output, output_type = capture_lhs(lhs)

    inputs = get_variables(rhs)
    rhs = variables_to_particle(inputs, rhs)

    sampler = (particle, rng) -> begin
        output_val = NamedTuple{(output,)}((eval(rhs),))
        return merge(particle, output_val)
    end

    weighter = nothing

    return FKStep(inputs, output, output_type, sampler, weighter)
end

function sampling_to_FKStep(expr)
    @capture(expr, lhs_ ~ f_(args__)) || error("Expression must be a statement `x ~ sampler(args)`")

    output, output_type = capture_lhs(lhs)

    inputs, args = process_args(args)

    sampler_code = quote
        let ws = $f  # This binds the instance WeightedSampler into the definition and allows the compiler to optimize out any allocations that would result from calling it from the outside.
            (particle, rng) -> begin
                sampled_value = ws.sampler($(args...), rng)
                output_val = NamedTuple{($(QuoteNode(output)),)}((sampled_value,))
                return merge(particle, output_val)
            end
        end
    end

    sampler = Core.eval(Main, sampler_code)

    weighter_code = quote
        let ws = $f
            (particle) -> begin
                return ws.weighter($(args...), particle[$(QuoteNode(output))])
            end
        end
    end

    weighter = Core.eval(Main, weighter_code)

    return FKStep(inputs, output, output_type, sampler, weighter)
end

function observe_to_FKStep(expr)
    @capture(expr, lhs_ -> f_(args__)) || error("Expression must be an observation `x -> sampler(args)`")


    lhs_vars = get_variables(lhs)
    lhs = variables_to_particle(lhs_vars, lhs)

    output = :nothing
    output_type = Nothing
    sampler = nothing

    inputs, args = process_args(args)

    weighter_code = quote
        let ws = $f
            (particle) -> begin
                if ws.weighter === nothing
                    return ws.logpdf($(args...), $lhs)
                else
                    return ws.logpdf($(args...), $lhs) + ws.weighter($(args...), $lhs)
                end
            end
        end
    end

    weighter = Core.eval(Main, weighter_code)

    return FKStep(inputs, output, output_type, sampler, weighter)
end

function extract_steps(body)
    steps = FKStep[]
    for expr in body
        if @capture(expr, _ = _)
            push!(steps, assignment_to_FKStep(expr))
        elseif @capture(expr, _ ~ f_(args__))
            push!(steps, sampling_to_FKStep(expr))
        elseif @capture(expr, _ -> f_(args__))
            push!(steps, observe_to_FKStep(expr))
        elseif @capture(expr, for loop_var_ in start_:stop_
            loop_body__
        end) ||
               @capture(expr, for loop_var_ = start_:stop_
            loop_body__
        end)

            for i in eval(start):eval(stop)
                map(loop_body) do step
                    processed_step = process_loop_statement(step, loop_var, i)
                    push!(steps, sampling_to_FKStep(processed_step))
                end
            end
        else
            error("ParseError: Expression must be a statement or loop")
        end
    end
    return steps
end


function simplify_arithmetic(expr)
    if expr isa Number
        return expr
    elseif !(expr isa Expr)
        return expr
    end

    # Recursively simplify arguments
    args = [simplify_arithmetic(arg) for arg in expr.args]

    # If all arguments are numbers, try to evaluate the expression
    if expr.head == :call && all(x -> x isa Number, args[2:end])
        try
            return eval(Expr(expr.head, args...))
        catch
            return Expr(expr.head, args...)
        end
    end

    return Expr(expr.head, args...)
end

function process_loop_statement(loop_body, loop_var, loop_value)
    MacroTools.postwalk(loop_body) do expr
        if expr isa Symbol && expr == loop_var
            return loop_value
        elseif @capture(expr, var_name_[index_])
            substituted_index = process_loop_statement(index, loop_var, loop_value)
            return simplify_arithmetic(expr)
        elseif @capture(expr, var_name_{index_})
            substituted_index = process_loop_statement(index, loop_var, loop_value)
            return simplify_arithmetic(expr)
        else
            return expr
        end
    end
end

macro model(expr)
    @capture(expr, function name_(args__)
            body__
        end) ||
        @capture(expr, function name_()
            body__
        end) ||
        error("Expression must be a function definition")

    param_names = [arg isa Symbol ? arg : arg.args[1] for arg in args]

    return esc(quote
        function $name($(args...))
            param_values = [$(param_names...)]

            # Substitute parameter values in the function body
            substituted_body = map($body) do expr
                $MacroTools.postwalk(expr) do x
                    (x isa Symbol && x in $param_names) ?
                    param_values[findfirst(==(x), $param_names)] : x
                end
            end

            steps = $DrawingInferences.extract_steps(substituted_body)

            return $DrawingInferences.FKModel(steps)
        end
    end)
end
