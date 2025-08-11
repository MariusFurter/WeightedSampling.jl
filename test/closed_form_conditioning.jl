using Revise

using DrawingInferences
using Distributions
using CairoMakie

### Beta-binomial model

@smc function beta_binomial(a)
    b = 1.0
    p ~ Beta(a, b)
    7 -> Binomial(10, p)
end

@time samples = beta_binomial(1.0; n_particles=100_000, ess_perc_min=0.5)

hist(samples[!, :p], weights=exp.(samples[!, :weights]), bins=50, normalization=:pdf)

#Closed form solution
function p_given_y(n, k, a, b)
    # n: number of trials
    # k: number of successes
    # a: alpha parameter of the Beta prior
    # b: beta parameter of the Beta prior
    # Returns the posterior probability of p given y
    return Beta(k + a, n - k + b)
end

xs = 0:0.01:1.0
lines!(xs, pdf.(p_given_y(10, 7, 1, 1), xs))
current_figure()