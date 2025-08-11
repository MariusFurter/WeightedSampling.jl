"""
Exponentiate and normalize the log weights in-place.
"""
function exp_norm_weights!(weights::Vector{Float64})
    max_weight = maximum(weights)
    weights .= exp.(weights .- max_weight)
    sum_exp_weights = sum(weights)
    weights ./= sum_exp_weights
    return nothing
end

"""
Exponentiate and normalize the log weights, returning a new vector.
"""
function exp_norm_weights(weights::Vector{Float64})
    max_weight = maximum(weights)
    exp_weights = exp.(weights .- max_weight)
    sum_exp_weights = sum(exp_weights)
    return exp_weights ./ sum_exp_weights
end


"""
Compute the weighted expectation of an expression `f` with respect to particle samples. Variable names in `f` that match column names in `particles` will reference the corresponding particle values.
"""
macro E(f, particles)
    f_quoted = QuoteNode(f)

    return esc(quote
        let p = $particles
            particle_names = Set(Symbol.(names(p)))
            f_replaced = DrawingInferences.replace_symbols_in($f_quoted, particle_names)
            weights = DrawingInferences.exp_norm_weights(p.weights)

            expectation_func = eval(quote
                (particles, w) -> sum($(f_replaced) .* w)
            end)

            expectation_func(p, weights)
        end
    end)
end
