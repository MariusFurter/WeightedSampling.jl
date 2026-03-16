"""
Hierarchical Regression Benchmark — NumPyro (NUTS)

Model:
  mu_alpha ~ N(0, 10), mu_beta ~ N(0, 10)
  sigma_alpha ~ Exp(0.2), sigma_beta ~ Exp(0.2)   [mean 5]
  alpha_j ~ N(mu_alpha, sigma_alpha)  for j = 1,...,J
  beta_j ~ N(mu_beta, sigma_beta)    for j = 1,...,J
  y_ij ~ N(alpha_j + beta_j * x_ij, 1.0)

J = 50 groups, N = 50 observations per group, 2500 total.
Compare with benchmarks/hierarchical_regression.jl

Timing fairness: reports first run (includes JIT) and subsequent runs (JIT cached)
separately, taking the median of cached runs.

Usage: python benchmarks/hierarchical_regression_numpyro.py
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_platform("cpu")

TRUE_MU_ALPHA = 2.0
TRUE_MU_BETA = -1.0
TRUE_SIGMA_ALPHA = 1.5
TRUE_SIGMA_BETA = 0.8
SIGMA_Y = 1.0


def generate_data(J, N, seed=42):
    np.random.seed(seed)
    alphas = TRUE_MU_ALPHA + TRUE_SIGMA_ALPHA * np.random.randn(J)
    betas  = TRUE_MU_BETA  + TRUE_SIGMA_BETA  * np.random.randn(J)

    x_all, y_all, group_all = [], [], []
    for j in range(J):
        xs = np.random.randn(N)
        ys = alphas[j] + betas[j] * xs + SIGMA_Y * np.random.randn(N)
        x_all.append(xs)
        y_all.append(ys)
        group_all.append(np.full(N, j))

    return (np.concatenate(x_all), np.concatenate(y_all),
            np.concatenate(group_all).astype(int), alphas, betas)


def hier_regression(J, x, y, group_idx):
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 10))
    mu_beta  = numpyro.sample("mu_beta", dist.Normal(0, 10))
    # Exponential(rate=0.2) has mean 5, matching Julia's Exponential(5)
    sigma_alpha = numpyro.sample("sigma_alpha", dist.Exponential(0.2))
    sigma_beta  = numpyro.sample("sigma_beta", dist.Exponential(0.2))

    with numpyro.plate("groups", J):
        alpha = numpyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))
        beta  = numpyro.sample("beta", dist.Normal(mu_beta, sigma_beta))

    mu = alpha[group_idx] + beta[group_idx] * x
    with numpyro.plate("obs", len(y)):
        numpyro.sample("y", dist.Normal(mu, SIGMA_Y), obs=y)


def run_benchmark(J, x, y, group_idx, true_alphas, true_betas,
                  num_warmup, num_samples, rng_key, label="", n_runs=3):
    x_jnp = jnp.array(x)
    y_jnp = jnp.array(y)
    group_jnp = jnp.array(group_idx)

    print(f"\n--- {label} (warmup={num_warmup}, samples={num_samples}) ---")

    nuts = NUTS(hier_regression)

    # Run 1: includes JIT compilation
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    start = time.perf_counter()
    mcmc.run(rng_key, J, x_jnp, y_jnp, group_jnp)
    jax.block_until_ready(mcmc.get_samples())
    time_with_jit = time.perf_counter() - start
    print(f"Time (incl JIT):    {time_with_jit:.4f} s")

    # Subsequent runs: JIT cached
    times = []
    for i in range(n_runs):
        key_i = random.fold_in(rng_key, i + 100)
        mcmc_i = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                      progress_bar=False)
        start = time.perf_counter()
        mcmc_i.run(key_i, J, x_jnp, y_jnp, group_jnp)
        samples = mcmc_i.get_samples()
        jax.block_until_ready(samples)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    med_time = np.median(times)
    print(f"Median time (cached): {med_time:.4f} s  "
          f"(range: {min(times):.4f} – {max(times):.4f})")

    # Posterior summaries
    mu_alpha = float(np.mean(samples["mu_alpha"]))
    mu_beta  = float(np.mean(samples["mu_beta"]))
    sigma_alpha = float(np.mean(samples["sigma_alpha"]))
    sigma_beta  = float(np.mean(samples["sigma_beta"]))

    alpha_means = np.mean(np.array(samples["alpha"]), axis=0)
    beta_means  = np.mean(np.array(samples["beta"]), axis=0)
    alpha_rmse = np.sqrt(np.mean((alpha_means - true_alphas)**2))
    beta_rmse  = np.sqrt(np.mean((beta_means - true_betas)**2))

    print(f"mu_alpha:           {mu_alpha:.3f}  (true: {TRUE_MU_ALPHA})")
    print(f"mu_beta:            {mu_beta:.3f}  (true: {TRUE_MU_BETA})")
    print(f"sigma_alpha:        {sigma_alpha:.3f}  (true: {TRUE_SIGMA_ALPHA})")
    print(f"sigma_beta:         {sigma_beta:.3f}  (true: {TRUE_SIGMA_BETA})")
    print(f"alpha RMSE:         {alpha_rmse:.4f}")
    print(f"beta RMSE:          {beta_rmse:.4f}")

    # Print NUTS diagnostics for the last run
    mcmc_i.print_summary()

    return time_with_jit, med_time


if __name__ == "__main__":
    J = 50
    N = 50

    x, y, group_idx, true_alphas, true_betas = generate_data(J, N)

    print("=" * 60)
    print("Hierarchical Regression — NumPyro (NUTS)")
    print("=" * 60)
    print(f"Groups: {J}, Obs/group: {N}, Total obs: {J * N}")
    print(f"True mu_alpha={TRUE_MU_ALPHA}, mu_beta={TRUE_MU_BETA}, "
          f"sigma_alpha={TRUE_SIGMA_ALPHA}, sigma_beta={TRUE_SIGMA_BETA}")

    rng_key = random.PRNGKey(0)

    # Standard run
    k1, k2 = random.split(rng_key)
    run_benchmark(J, x, y, group_idx, true_alphas, true_betas,
                  500, 2000, k1, label="Standard (500+2000)")

    # Large run
    run_benchmark(J, x, y, group_idx, true_alphas, true_betas,
                  1000, 5000, k2, label="Large (1000+5000)")

    print("\n" + "=" * 60)
