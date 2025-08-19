# Rewrite sampling and observe statements as follows:

# A better version would directly compute the difference of the proposal and current value logpdf. This would allow to not compute increments where both input and outputs have the same values.


# Sampling: y ~ f_(args...)
# 1. Extract lhs and rhs variables.
# 2. append the following if current[y, args...] != proposed[y, args...]:
# if tracker < target_depth
#   scores .+= f.lpdf.( proposed_replaced_args..., particles[!, lhs] )
#   scores .-= f.lpdf.( current_replaced_args..., particles[!, lhs] )
# else
#   return scores

# Observe: 5 => f_(args...)
# 1. Extract rhs variables.
# 2. append the following if current[y, args...] != proposed[y, args...]
# if tracker < target_depth:
#   scores .+= f.lpdf.( proposed_replaced_args..., lhs) 
#   scores .-= f.lpdf.(current_replaced_args..., lhs )
# else
#   return scores

# Target depth should be incremented in smc body each time a sampling or observe statement is encountered. It can then be used as the argument in the rewritten move statements.

# Bonus: Add check that the same variable is not used multiple times. E.g keep a list of variables used in sampling statements up until now and check that the current variable is not in that list.

function build_logpdf_difference(body)
    return quote
        function logpdf_difference(current_particles, proposed_particles, target_depth)
            scores = zeros(length(current_particles))
            # Compute positions where current and proposed particles differ
            tracker = 0
            $logpdf_code
        end
    end
end

## Proposal kernel should take particles and var_symbols as input
## mutate the particles in place, and return a vector of log scores.
## (particles, in_symbols, out_symbols) -> log_scores

## This is general enough to allow for adaptive proposals.
## In fact, one could also use this signature for SMCKernels.
## No: does not seem to support partial evaluation with literals well.

## Syntax for moves:
function bla()
    x ~ Normal(0, 1)
    x << RW()
    RW!(x)
    move!(x)
    y ~ Normal(x, 1)
    3 => Normal(y, 1)
    u = sin(x)
end

e = :(x << AdaptiveNormal(x))
dump(e)