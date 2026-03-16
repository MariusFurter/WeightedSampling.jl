"""
State-Space Model Benchmark — NumPyro (NUTS)

Linear-Gaussian SSM:
  x_0 ~ N(0, 1)
  x_t = 0.9 * x_{t-1} + eps_t,  eps_t ~ N(0, 1)
  y_t ~ N(x_t, 1)

NUTS performs batch smoothing over all T latent states.
Uses numpyro.contrib.control_flow.scan for efficient JIT compilation.
Compare with benchmarks/ssm_comparison.jl (online SMC filtering).

Timing fairness: reports first run (includes JIT) and subsequent runs (JIT cached)
separately, taking the median of cached runs.

Usage: python benchmarks/ssm_comparison_numpyro.py
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.control_flow import scan

numpyro.set_platform("cpu")

A = 0.9
Q = 1.0
R = 1.0


def generate_data(T, seed=42):
    np.random.seed(seed)
    xs = [0.0]
    ys = []
    for t in range(T):
        x_new = A * xs[-1] + Q * np.random.randn()
        xs.append(x_new)
        ys.append(x_new + R * np.random.randn())
    return np.array(xs), np.array(ys)


def kalman_filter(ys):
    T = len(ys)
    mu = 0.0
    sig2 = 1.0
    log_evidence = 0.0
    filtered_means = []

    for t in range(T):
        mu_pred = A * mu
        sig2_pred = A**2 * sig2 + Q**2
        S = sig2_pred + R**2
        K = sig2_pred / S
        innov = ys[t] - mu_pred
        mu = mu_pred + K * innov
        sig2 = (1 - K) * sig2_pred
        log_evidence += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)
        filtered_means.append(mu)

    return filtered_means, log_evidence


# --- SSM model using scan for efficient JIT ---
def transition(x_prev, y_t):
    x = numpyro.sample("x", dist.Normal(A * x_prev, Q))
    numpyro.sample("y", dist.Normal(x, R), obs=y_t)
    return x, x


def ssm_model(y):
    x_0 = numpyro.sample("x_0", dist.Normal(0.0, 1.0))
    _, xs = scan(transition, x_0, y)


def run_benchmark(y, kf_means, kf_evidence, num_warmup, num_samples,
                  rng_key, label="", n_runs=3):
    T = len(y)
    y_jnp = jnp.array(y)

    print(f"\n--- {label} (warmup={num_warmup}, samples={num_samples}) ---")

    nuts = NUTS(ssm_model)

    # Run 1: includes JIT compilation
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    start = time.perf_counter()
    mcmc.run(rng_key, y_jnp, extra_fields=("potential_energy",))
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
        mcmc_i.run(key_i, y_jnp, extra_fields=("potential_energy",))
        samples = mcmc_i.get_samples()
        jax.block_until_ready(samples)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    med_time = np.median(times)
    print(f"Median time (cached): {med_time:.4f} s  "
          f"(range: {min(times):.4f} – {max(times):.4f})")

    # Posterior on last state
    x_samples = np.array(samples["x"])
    x_T_samples = x_samples[:, -1]
    print(f"E[x_T]:             {x_T_samples.mean():.4f}  "
          f"(Kalman: {kf_means[-1]:.4f})")
    print(f"Std[x_T]:           {x_T_samples.std():.4f}")

    # Log joint (not evidence, but useful for diagnostics)
    pe = mcmc_i.get_extra_fields()["potential_energy"]
    log_joint = float(np.mean(-pe))
    print(f"E[log joint]:       {log_joint:.4f}")
    print(f"Kalman log Z:       {kf_evidence:.4f}")

    return time_with_jit, med_time


if __name__ == "__main__":
    T = 200

    xs_true, ys = generate_data(T)
    kf_means, kf_evidence = kalman_filter(ys)

    print("=" * 60)
    print("SSM Benchmark — NumPyro (NUTS via scan)")
    print("=" * 60)
    print(f"Timesteps: {T}")
    print(f"Kalman log evidence: {kf_evidence:.4f}")

    rng_key = random.PRNGKey(0)

    # Small run
    k1, k2 = random.split(rng_key)
    run_benchmark(ys, kf_means, kf_evidence, 200, 500, k1,
                  label="Small (200+500)", n_runs=3)

    # Standard run
    run_benchmark(ys, kf_means, kf_evidence, 500, 2000, k2,
                  label="Standard (500+2000)", n_runs=3)

    # Large run
    k3, _ = random.split(k2)
    run_benchmark(ys, kf_means, kf_evidence, 1000, 5000, k3,
                  label="Large (1000+5000)", n_runs=3)

    print("\n" + "=" * 60)
