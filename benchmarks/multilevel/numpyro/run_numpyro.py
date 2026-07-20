"""
run_numpyro.py

Loads a dataset written by `../simulate.jl`, runs NUTS on the hierarchical
regression model (`model.py`), and prints timing + inference-quality metrics
as `key=value` lines to stdout (same format as `../WeightedSampling/run_ws.jl`,
for easy consumption by `../run_benchmark.py`).

Also reports CPU time / effective sample, using arviz's `ess` (bulk ESS,
rank-normalized-split estimator) computed over all scalar sites (`mu_alpha`,
`tau_alpha`, `beta`, `sigma`, and each component of `alpha`). NOTE: arviz's
ESS estimator can exceed `num_samples` (e.g. under anti-correlated/antithetic
sampling) -- this is expected, not a bug, so `ess_min`/`ess_mean` are reported
as-is without clipping.

Usage: python run_numpyro.py <data_dir> [--num_warmup 500] [--num_samples 1000] [--seed 0]
"""
import argparse
import os
import time

import arviz as az
import jax
import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS

from model import model


def read_data(data_dir):
    path = os.path.join(data_dir, "data.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    group = data[:, 0].astype(int) - 1  # 0-indexed for numpyro
    x = data[:, 1]
    y = data[:, 2]
    J = int(group.max()) + 1
    return J, group, x, y


def read_true_params(data_dir):
    truth = {}
    with open(os.path.join(data_dir, "true_params.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split("=")
            truth[k] = float(v)
    return truth


def run_numpyro_benchmark(data_dir, num_warmup=500, num_samples=1000, seed=0):
    J, group_idx, x, y = read_data(data_dir)
    truth = read_true_params(data_dir)
    true_alpha = np.array([truth[f"alpha_{j+1}"] for j in range(J)])
    n_obs = len(x) // J

    rng_key = random.key(seed)

    # Warm-up run (tiny, same argument shapes as the sampling sites so JAX's
    # jit cache is reused) to exclude JIT compilation from the timed run.
    warmup_group = np.array([0])
    warmup_x = np.array([0.0])
    warmup_y = np.array([0.0])
    warmup_mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5, progress_bar=False)
    warmup_mcmc.run(rng_key, 1, warmup_group, warmup_x, y=warmup_y)
    jax.block_until_ready(warmup_mcmc.get_samples())

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)

    start = time.perf_counter()
    mcmc.run(rng_key, J, group_idx, x, y=y)
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    elapsed = time.perf_counter() - start

    alpha_est = np.asarray(samples["alpha"]).mean(axis=0)
    mu_alpha_est = float(np.asarray(samples["mu_alpha"]).mean())
    tau_alpha_est = float(np.asarray(samples["tau_alpha"]).mean())
    beta_est = float(np.asarray(samples["beta"]).mean())
    sigma_est = float(np.asarray(samples["sigma"]).mean())

    rmse_alpha = float(np.sqrt(np.mean((alpha_est - true_alpha) ** 2)))

    # Effective sample size (arviz bulk ESS) over every scalar site, flattened
    # across the vector-valued `alpha` site's J components. Computed from the
    # same `samples` used above (single chain; arviz adds the chain dim).
    idata = az.from_numpyro(mcmc)
    ess_ds = az.ess(idata)
    ess_values = np.concatenate([np.atleast_1d(ess_ds[var].values).ravel() for var in ess_ds.data_vars])
    ess_min = float(np.min(ess_values))
    ess_mean = float(np.mean(ess_values))

    print(f"J={J}")
    print(f"n_obs={n_obs}")
    print(f"num_warmup={num_warmup}")
    print(f"num_samples={num_samples}")
    print(f"elapsed_time={elapsed:.6f}")
    print(f"ess_min={ess_min:.4f}")
    print(f"ess_mean={ess_mean:.4f}")
    print(f"time_per_ess_min={elapsed / ess_min:.6f}")
    print(f"time_per_ess_mean={elapsed / ess_mean:.6f}")
    print(f"rmse_alpha={rmse_alpha:.6f}")
    print(f"mu_alpha_est={mu_alpha_est:.6f}")
    print(f"mu_alpha_err={abs(mu_alpha_est - truth['mu_alpha']):.6f}")
    print(f"tau_alpha_est={tau_alpha_est:.6f}")
    print(f"tau_alpha_err={abs(tau_alpha_est - truth['tau_alpha']):.6f}")
    print(f"beta_est={beta_est:.6f}")
    print(f"beta_err={abs(beta_est - truth['beta']):.6f}")
    print(f"sigma_est={sigma_est:.6f}")
    print(f"sigma_err={abs(sigma_est - truth['sigma']):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--num_warmup", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_numpyro_benchmark(args.data_dir, args.num_warmup, args.num_samples, args.seed)
