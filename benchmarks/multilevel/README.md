# Hierarchical regression benchmark: WeightedSampling vs NumPyro

Compares WeightedSampling (SMC) against NumPyro (NUTS) on a hierarchical
(partial-pooling) linear regression model, generalizing the "Eight Schools"
example from the [NumPyro getting-started guide](https://num.pyro.ai/en/stable/getting_started.html)
to `J` groups with `n_obs` regression observations each:

```
mu_alpha  ~ Normal(0, 10)
tau_alpha ~ Exponential(1)
beta      ~ Normal(0, 10)
sigma     ~ Exponential(1)
alpha[j]  ~ Normal(mu_alpha, tau_alpha)                  for j in 1:J
y[i]      ~ Normal(alpha[group[i]] + beta * x[i], sigma)
```

## Layout

- `simulate.jl` — simulates ground truth data (shared by both implementations)
  into `data/<config>/data.csv` + `true_params.txt`.
- `WeightedSampling/` — `model.jl` (the `@model`), `run_ws.jl` (CLI runner),
  own `Project.toml` with `WeightedSampling` added as a dev dependency.
- `numpyro/` — `model.py`, `run_numpyro.py` (CLI runner), and a `venv/`
  (created via `python3 -m venv venv && pip install numpyro pandas`).
- `run_benchmark.py` — orchestrates both across a grid of `(J, n_obs)`
  configs. For each config: simulates data, runs NumPyro once (fixed
  `num_warmup`/`num_samples`), then **calibrates** WeightedSampling's
  particle count `N` (doubling from `N_START` until its RMSE of posterior
  mean `alpha_j` vs. ground truth is within `TOLERANCE` of NumPyro's RMSE, or
  `N_MAX` is reached), so both are compared at roughly matched inference
  quality.

## Running

```bash
python3 run_benchmark.py
```

Both runners can also be invoked directly for a single dataset:

```bash
julia --project=. simulate.jl 20 10 42 data/J20_n10
julia --project=WeightedSampling WeightedSampling/run_ws.jl data/J20_n10 8000
numpyro/venv/bin/python numpyro/run_numpyro.py data/J20_n10 --num_warmup 500 --num_samples 1000
```

Each runner prints `key=value` lines (elapsed time, allocations where
applicable, RMSE of `alpha` vs. ground truth, and point estimates/errors for
the global parameters), plus an effective-sample-size-based metric:

- **NumPyro** (`ess_min`, `ess_mean`, `time_per_ess_min`, `time_per_ess_mean`):
  computed via `arviz.ess` (bulk ESS, rank-normalized-split estimator) over
  every scalar site (`mu_alpha`, `tau_alpha`, `beta`, `sigma`, and each
  component of `alpha`). **Note:** arviz's ESS estimator can exceed
  `num_samples` under favorable/antithetic mixing — this is expected, not a
  bug.
- **WeightedSampling** (`ess`, `time_per_ess`): the standard SMC particle
  effective sample size `1 / sum(w_i^2)` for normalized final weights `w`
  (capped at `N`).

These two ESS notions are conceptually different (MCMC draws vs. weighted
particles) so "CPU time per effective sample" should be read as a rough proxy
for each method's own internal efficiency, not an exact apples-to-apples
statistic across methods.
