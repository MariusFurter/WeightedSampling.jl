"""
Exponentiate and normalize the log weights in-place.
"""
function exp_norm!(weights::Vector{Float64})
    max_weight = maximum(weights)
    weights .= exp.(weights .- max_weight)
    sum_exp_weights = sum(weights)
    weights ./= sum_exp_weights
    return nothing
end

"""
Exponentiate and normalize the log weights, returning a new vector.
"""
function exp_norm(weights::Vector{Float64})
    max_weight = maximum(weights)
    exp_weights = exp.(weights .- max_weight)
    sum_exp_weights = sum(exp_weights)
    return exp_weights ./ sum_exp_weights
end

expectation(values, weights) = sum(values .* exp_norm(weights))


"""
Compute the weighted expectation of an anonymous function `f` with respect to particle samples. Argument names of `f` reference the corresponding particle values.
"""
macro E(f, particles)
    if @capture(f, arg_Symbol -> body_)
        args = [arg]
    else
        @capture(f, (args__,) -> body_) ||
            error("E macro requires a function of the form (args...) -> expr")
    end

    body_replaced = WeightedSampling.replace_symbols_in(body, args, :particles, :N)
    return esc(quote
        let particles = $particles
            N = WeightedSampling.nrow(particles)
            weights = WeightedSampling.exp_norm(particles.weights)
            values = $body_replaced
            sum(values .* weights)
        end
    end)
end


"""
Compute the weighted expectation of an expression `f` with respect to particle samples. Variable names in `f` that match column names in `particles` will reference the corresponding particle values.
"""
macro E_old(f, particles)
    f_quoted = QuoteNode(f)

    return esc(quote
        let particles = $particles
            particle_names = Set(Symbol.(names(particles)))
            f_replaced = WeightedSampling.replace_symbols_in($f_quoted, particle_names, :particles, :N)
            weights = WeightedSampling.exp_norm(particles.weights)

            f_func_code = quote
                (particles) -> $f_replaced
            end
            f_func = eval(f_func_code)

            f_eval = Base.invokelatest(f_func, particles)

            sum(f_eval .* weights)
        end
    end)
end

"""
Compute the weighted expectation of an expression `f` with respect to particle samples. Variable names in `f` not in `exceptions` are interpreted as columns in `particles`.
"""
macro E_except(f, particles, exceptions=Symbol[])
    exceptions = Set(Symbol.(eval(exceptions)))
    f_replaced = replace_symbols_except(f, exceptions, :particles, :N)
    return esc(
        quote
            let particles = $particles
                weights = WeightedSampling.exp_norm(particles.weights)
                sum($f_replaced .* weights)
            end
        end
    )
end
