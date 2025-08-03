using MacroTools

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
        return QuoteNode(expr)
    elseif @capture(expr, output_symbol_{index_})
        return :(Symbol($(QuoteNode(output_symbol)), $index))
    else
        error("Left-hand side must be a variable name `x` or an indexed variable `x{i}`")
    end
end

function rewrite_assignment(expr, particles_sym, exceptions)
    @capture(expr, lhs_ = rhs_)

    output = capture_lhs(lhs)

    particle_sym = gensym("particle")
    rhs_rewritten = replace_symbols(rhs, particle_sym, exceptions)

    quote
        for $particle_sym in $particles_sym.samples
            val = $(rhs_rewritten)
            setproperty!($particle_sym, $output, val)
        end

    end
end

struct SMCParticles{P}
    samples::Vector{P}
    weights::Vector{Float64}
end

mutable struct XYParticle
    x::Float64
    y1::Float64
    y2::Float64
end

macro parse(expr)
    @capture(expr, function name_()
        body__
    end) || error("Expression must be a function definition")

    function parse_body(body, exceptions)
        code = quote end
        for statement in body

            if @capture(statement, lhs_ = rhs_)
                rewritten_statement = rewrite_assignment(statement, :particles, exceptions)
                append!(code.args, rewritten_statement.args)

            elseif @capture(statement, for loop_var_ in start_:stop_
                loop_body__
            end)

                push!(exceptions, loop_var)
                e = quote
                    for $loop_var in $start:$stop
                        $(parse_body(loop_body, exceptions))
                    end
                end
                delete!(exceptions, loop_var)
                append!(code.args, e.args)
            end
        end
        return code
    end

    exceptions = Set{Symbol}()
    parse_body(body, exceptions)
end


particles = SMCParticles{XYParticle}([XYParticle(0.0, 0.0, 0.0) for _ in 1:1000], [0.0 for _ in 1:1000])

## Assignments handled

@parse function bla()
    for j in 1:2
        for i in 1:2
            y{i} = x + j
        end
    end
end

