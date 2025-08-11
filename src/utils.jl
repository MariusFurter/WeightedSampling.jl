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
Compute (weighted) expectation of the expression `f` wrt particles. The variable names in `f` are interpreted as columns in the `particles` DataFrame.
"""
macro E(f, particles)
    f_replaced = replace_symbols(f, Set{Symbol}())
    return quote
        let particles = $(esc(particles))
            weights = exp_norm_weights(particles.weights)
            sum($f_replaced .* weights)
        end
    end
end
