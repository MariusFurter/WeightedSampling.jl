### Beta-binomial model where k out of n trials are successes, with T total trials.
function beta_binomial_test(n, k, T, a, b)
    Random.seed!(42)

    @smc function beta_binomial(n, k, T, a, b)
        a = a
        b = b
        p ~ Beta(a, b)
        for t in 1:T
            k -> Binomial(n, p)
        end
    end

    samples = beta_binomial(n, k, T, a, b; n_particles=100_000, ess_perc_min=0.0)

    smc_val = @E(p, samples)

    #Closed form solution
    function p_given_y(n, k, T, a, b)
        return Beta(T * k + a, T * (n - k) + b)
    end

    exact_val = mean(p_given_y(n, k, T, a, b))

    isapprox(smc_val, exact_val, atol=1e-2)
end