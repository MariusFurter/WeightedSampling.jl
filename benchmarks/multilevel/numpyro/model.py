"""
model.py

Hierarchical (partial-pooling) linear regression model in NumPyro, matching
`../WeightedSampling/model.jl`:

    mu_alpha ~ Normal(0, 10)
    tau_alpha ~ Exponential(1)
    beta ~ Normal(0, 10)
    sigma ~ Exponential(1)
    alpha[j] ~ Normal(mu_alpha, tau_alpha)                for j in 1:J
    y[i] ~ Normal(alpha[group[i]] + beta * x[i], sigma)
"""
import numpyro
import numpyro.distributions as dist


def model(J, group_idx, x, y=None):
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 10.0))
    tau_alpha = numpyro.sample("tau_alpha", dist.Exponential(1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    with numpyro.plate("J", J):
        alpha = numpyro.sample("alpha", dist.Normal(mu_alpha, tau_alpha))

    mu = alpha[group_idx] + beta * x
    with numpyro.plate("N", len(x)):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
