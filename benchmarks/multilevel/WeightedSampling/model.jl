# model.jl
#
# Hierarchical (partial-pooling) linear regression model, `@model`-style,
# mirroring `examples/eight_schools.jl` but generalized to `J` groups each with
# `n_obs` observations and a regression predictor `x`:
#
#     mu_alpha ~ Normal(0, 10)
#     tau_alpha ~ Exponential(1)
#     beta ~ Normal(0, 10)
#     sigma ~ Exponential(1)
#     alpha[j] ~ Normal(mu_alpha, tau_alpha)                    for j in 1:J
#     y ~ Normal(alpha[j] + beta * x, sigma)   for each (x, y) observation in group j
#
# Global parameters (`mu_alpha`, `tau_alpha`, `beta`, `sigma`) are refreshed via
# `autoRW` Metropolis-Hastings moves after each group's observations have been
# incorporated and the particle set resampled (same pattern as
# `examples/eight_schools.jl`).
using WeightedSampling

@model function hierarchical_regression(J, groups)
    mu_alpha ~ Normal(0.0, 10.0)
    tau_alpha ~ Exponential(1.0)
    beta ~ Normal(0.0, 10.0)
    sigma ~ Exponential(1.0)
    alpha .= zeros(J)
    for j in 1:J
        alpha[j] ~ Normal(mu_alpha, tau_alpha)
        obs = groups[j]
        for (x, y) in obs
            y => Normal(alpha[j] + beta * x, sigma)
        end
        if resampled
            mu_alpha << autoRW()
            tau_alpha << autoRW(1e-3, (0.0, Inf))
            beta << autoRW()
            sigma << autoRW(1e-3, (0.0, Inf))
        end
    end
end
