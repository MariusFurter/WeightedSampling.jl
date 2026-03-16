"""
Eight Schools Benchmark — NumPyro (NUTS)

Hierarchical model from Gelman et al. (2003) "Bayesian Data Analysis", Sec 5.5.
Compare with benchmarks/eight_schools.jl (SMC via WeightedSampling.jl).

Usage: python benchmarks/eight_schools_numpyro.py
Requires: pip install numpyro jax jaxlib
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam

numpyro.set_platform("cpu")

# ---- Data ----
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


# ---- Centered parameterization ----
def eight_schools_centered(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


# ---- Non-centered parameterization ----
def eight_schools_noncentered(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0),
                    dist.transforms.AffineTransform(mu, tau),
                ),
            )
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def run_benchmark(model_fn, model_name, num_warmup, num_samples, rng_key):
    print(f"\n--- {model_name} (warmup={num_warmup}, samples={num_samples}) ---")

    nuts_kernel = NUTS(model_fn)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

    start = time.perf_counter()
    mcmc.run(rng_key, J, sigma, y=y, extra_fields=("potential_energy",))
    elapsed = time.perf_counter() - start

    # Posterior summaries
    samples = mcmc.get_samples()
    mu_samples = np.array(samples["mu"])
    tau_samples = np.array(samples["tau"])
    theta_samples = np.array(samples["theta"])

    pe = mcmc.get_extra_fields()["potential_energy"]
    log_joint = float(np.mean(-pe))

    print(f"Time:         {elapsed:.4f} s")
    print(f"E[log joint]: {log_joint:.4f}")
    print(f"mu:           {mu_samples.mean():.2f} ± {mu_samples.std():.2f}")
    print(f"tau:          {tau_samples.mean():.2f} ± {tau_samples.std():.2f}")
    theta_means = theta_samples.mean(axis=0)
    print(f"theta means:  {', '.join(f'{m:.2f}' for m in theta_means)}")

    n_eff = mcmc.get_extra_fields().get("num_steps", None)
    mcmc.print_summary()

    return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Eight Schools Benchmark — NumPyro (NUTS)")
    print("=" * 60)

    rng_key = random.PRNGKey(0)

    # Centered
    k1, k2 = random.split(rng_key)
    run_benchmark(eight_schools_centered, "Centered", 500, 1000, k1)

    # Non-centered
    run_benchmark(eight_schools_noncentered, "Non-centered", 500, 1000, k2)

    # Larger run (non-centered)
    k3, _ = random.split(k2)
    run_benchmark(eight_schools_noncentered, "Non-centered (large)", 1000, 5000, k3)

    print("\n" + "=" * 60)
