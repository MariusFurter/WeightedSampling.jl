using MacroTools

function parse_variables(expr)
    MacroTools.@capture(expr, (vars__,)) && return Symbol[v for v in vars if v isa Symbol]
    return expr isa Symbol ? Symbol[expr] : Symbol[expr]
end

function parse_assignment(assignment)
    MacroTools.@capture(assignment, lhs_ = rhs_) || error("Expected assignment expression")
    outputs = parse_variables(lhs)

    if MacroTools.@capture(rhs, func_(args__))
        inputs = mapreduce(parse_variables, vcat, args, init=Symbol[])
        return :(FKStep($inputs, $outputs, $func))
    elseif rhs isa Symbol
        return :(FKStep(Symbol[$rhs], $outputs, identity))
    else
        error("Unsupported right-hand side expression: $rhs")
    end
end

function extract_assignments(body)
    assignments = []
    for expr in body
        if MacroTools.@capture(expr, _ = _)
            push!(assignments, expr)
        elseif MacroTools.@capture(expr, for _var_ in start_:stop_
            loop_body__
        end) ||
               MacroTools.@capture(expr, for _var_ = start_:stop_
            loop_body__
        end)
            loop_assignments = filter(ex -> MacroTools.@capture(ex, _ = _), loop_body)
            for _ in eval(start):eval(stop)
                append!(assignments, loop_assignments)
            end
        end
    end
    return assignments
end

# Macro to parse DSL expressions and generate FKModel
macro fk(expr)
    MacroTools.@capture(expr, function name_(args__)
            body__
        end) ||
        MacroTools.@capture(expr, function name_()
            body__
        end) ||
        error("@fk macro expects a function definition")

    param_names = [arg isa Symbol ? arg : arg.args[1] for arg in args]

    return esc(quote
        function $name($(args...))
            param_values = [$(param_names...)]

            substituted_body = map($body) do expr
                $MacroTools.postwalk(expr) do x
                    (x isa Symbol && x in $param_names) ?
                    param_values[findfirst(==(x), $param_names)] : x
                end
            end

            assignments = $DrawingInferences.extract_assignments(substituted_body)
            steps = [eval($DrawingInferences.parse_assignment(assignment)) for assignment in assignments]
            return $DrawingInferences.FKModel((steps...,))
        end
    end)
end