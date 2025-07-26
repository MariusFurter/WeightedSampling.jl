using MacroTools

function extract_assignments(body)
    assignments = []
    for expr in body
        if @capture(expr, _ = _)
            push!(assignments, expr)
        elseif @capture(expr, for loop_var_ in start_:stop_
            loop_body__
        end) ||
               @capture(expr, for loop_var_ = start_:stop_
            loop_body__
        end)
            # Extract assignments from loop body and expand for each iteration
            loop_assignments = filter(ex -> @capture(ex, _ = _), loop_body)
            for i in eval(start):eval(stop)
                # For each iteration, substitute the loop variable and evaluate curly brace expressions
                substituted_assignments = map(loop_assignments) do assignment
                    interpolate_loop_braces(assignment, loop_var, i)
                end
                append!(assignments, substituted_assignments)
            end
        else
            error("ParseError: Expression must be an assignment or loop")
        end
    end
    return assignments
end

function interpolate_loop_braces(expr, loop_var, loop_value)
    MacroTools.postwalk(expr) do x
        if x isa Symbol && x == loop_var
            # Replace loop variable with its current value
            return loop_value
        elseif @capture(x, var_name_{index_})
            # Handle variable names with curly braces like x_{i} or x_{i-1}
            # First substitute the loop variable in the index expression
            substituted_index = MacroTools.postwalk(index) do y
                (y isa Symbol && y == loop_var) ? loop_value : y
            end
            # Evaluate the index expression
            evaluated_index = eval(substituted_index)
            # Create new symbol with evaluated index
            return Symbol("$(var_name)$(evaluated_index)")
        else
            return x
        end
    end
end

function fkstep(assignment)
    @capture(assignment, lhs_ = rhs_) || error("Expression must be assignment")
    @capture(rhs, func_(args__)) || error("Right-hand side must be a function call")

    # Enforce single variable on left side
    @capture(lhs, output_var_symbol_) || error("Left-hand side must be a single variable name")
    isa(output_var_symbol, Symbol) || error("Left-hand side must be a single variable name")

    output_var = QuoteNode(output_var_symbol)

    # Convert args to symbols for inputs
    input_vars = tuple(args...)

    # Return the expression that creates an FKStep
    return quote
        let ops = $func
            # Create particle sampler function
            particle_sampler = (particle, rng) -> begin
                sampled_value = ops.sampler($((:(particle.$var) for var in input_vars)...), rng)
                output = NamedTuple{($output_var,)}((sampled_value,))
                return merge(particle, output)
            end

            # Create particle weighter function  
            particle_weighter = (particle) -> begin
                return ops.weighter($((:(particle.$var) for var in input_vars)...), particle.$(output_var_symbol))
            end

            # Get output type from the function
            the_output_type = ops.output_type()

            FKStep($input_vars, $output_var, the_output_type,
                particle_sampler, particle_weighter)
        end
    end
end

# Keep @fkstep as a convenience macro for direct use
macro fkstep(expr)
    return esc(fkstep(expr))
end

macro fk(expr)
    @capture(expr, function name_(args__)
            body__
        end) ||
        @capture(expr, function name_()
            body__
        end) ||
        error("@fk expects a function definition")

    param_names = [arg isa Symbol ? arg : arg.args[1] for arg in args]

    ### Validation: function body must be well-formed Julia code
    try
        # Create a temporary function definition to test syntax
        temp_func = quote
            function $(gensym("temp_$(name)_validation"))($(args...))
                $(body...)
            end
        end
        eval(temp_func)
    catch e
        error("@fk input is not well-formed Julia code: $e")
    end

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

            # Extract assignments from the substituted body
            assignments = $DrawingInferences.extract_assignments(substituted_body)

            # Convert each assignment to an FKStep using fkstep function
            steps = [eval($DrawingInferences.fkstep(assignment)) for assignment in assignments]

            return $DrawingInferences.FKModel(tuple(steps...))
        end
    end)
end