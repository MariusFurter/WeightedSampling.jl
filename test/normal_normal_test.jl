### Normal-Normal conjugate model
# Prior: μ ~ Normal(0, σ_prior)
# Likelihood: data_val ~ Normal(μ, σ_obs)
# Posterior: Normal(post_mean, post_std) with closed-form solution
#
# Ported from WeightedSampling.torch test_simple.py

function normal_normal_test(; n_particles=100_000, atol_mean=0.1, atol_std=0.1)
    Random.seed!(42)

    prior_mean = 0.0
    prior_std = 10.0
    obs_std = 1.0
    data_val = 5.0

    @model function normal_normal_model(data_val, prior_mean, prior_std, obs_std)
        prior_mean = prior_mean
        prior_std = prior_std
        obs_std = obs_std
        μ ~ Normal(prior_mean, prior_std)
        data_val => Normal(μ, obs_std)
    end

    particles, evidence = normal_normal_model(data_val, prior_mean, prior_std, obs_std;
        n_particles=n_particles, show_progress=false)

    smc_mean = @E(μ -> μ, particles)
    smc_std = sqrt(@E(μ -> μ^2, particles) - smc_mean^2)

    # Closed-form posterior
    prior_var = prior_std^2
    obs_var = obs_std^2
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + data_val / obs_var)
    post_std = sqrt(post_var)

    mean_ok = isapprox(smc_mean, post_mean, atol=atol_mean)
    std_ok = isapprox(smc_std, post_std, atol=atol_std)

    return mean_ok && std_ok
end
