### Beta-binomial model where k out of n trials are successes, with T total trials.
function beta_binomial_test(n, k, T, a, b)
    Random.seed!(42)

    @smc function beta_binomial(n, k, T, a, b)
        a = a
        b = b
        p ~ Beta(a, b)
        for t in 1:T
            k => Binomial(n, p)
        end
    end

    samples, evidence = beta_binomial(n, k, T, a, b; n_particles=100_000)

    smc_p_val = @E(p -> p, samples)

    #Closed form solution
    function p_given_y(n, k, T, a, b)
        return Beta(T * k + a, T * (n - k) + b)
    end

    exact_p_val = mean(p_given_y(n, k, T, a, b))

    arbitrary_p_val = 0.5
    exact_evidence = logpdf(Beta(a, b), arbitrary_p_val) + sum(logpdf(Binomial(n, arbitrary_p_val), k) for _ in 1:T) - logpdf(p_given_y(n, k, T, a, b), arbitrary_p_val)

    isapprox(smc_p_val, exact_p_val, atol=1e-2) && isapprox(evidence, exact_evidence, atol=1e-1)
end