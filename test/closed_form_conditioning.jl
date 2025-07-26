using Distributions

### Beta-binomial model

@fk function beta_binomial()
    p = BetaSampler(1.0, 1.0)
    y = BinomialSampler(10, p)
end

#Closed form solution
function p_given_y(n, k, a, b)
    # n: number of trials
    # k: number of successes
    # a: alpha parameter of the Beta prior
    # b: beta parameter of the Beta prior
    # Returns the posterior probability of p given y
    return (k + a - 1) / (n + a + b - 2)
end