# Unified 1D linear-Gaussian SSM benchmark: WeightedSampling vs SequentialMonteCarlo.jl vs libbi

Compares 3 bootstrap-particle-filter implementations of the same model:

    x(0) ~ Normal(0, x0_std)
    x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
    y(t) ~ Normal(x(t), r)

with a=0.9, q=1.0, r=0.5, x0_std=1.0 (q/r are STD, not variance).

Each framework simulates its own observations from the same model/params
(matching T and RNG seed where applicable) rather than sharing bit-identical
data — see `simulate.jl`. All 3 frameworks are configured to resample at
**every step** (forced via `ess_perc_min=1.0` / `essThreshold=1.0` /
`ESS_REL=1.0`) to reduce cross-run variance and make libbi's per-update CPU
time directly comparable to the Julia frameworks.

## Layout

- `simulate.jl`: shared data-generation (`simulate_lgssm1d`) and exact
  Kalman-filter reference (`kalman_filter_evidence`), used by the two Julia
  frameworks below.
- `WeightedSampling/lgssm1d.jl`: `@model`-based implementation. Run:
  `julia --project=benchmarks/ssm/WeightedSampling -t 1 benchmarks/ssm/WeightedSampling/lgssm1d.jl [T] [N]`
- `SequentialMonteCarlo/lGModel.jl`: low-level `SMCModel`/`SMCIO` gold-standard
  baseline. Run:
  `julia --project=benchmarks/ssm/SequentialMonteCarlo -t 1 benchmarks/ssm/SequentialMonteCarlo/lGModel.jl [T] [N]`
  (must run single-threaded — RNGPool requirement).
- `libbi/lgssm1d/`: LibBi model + `run_pf.sh` runner (see its own README).
- `bench_single_update/bench_single_update.jl`: BenchmarkTools-based
  isolation of the cost of ONE mutate+observe+resample update, for
  WeightedSampling and SequentialMonteCarlo.jl (libbi has no equivalent
  isolated-step API, so it's excluded from this particular comparison — use
  the grid's per-`T`-scaling instead as a proxy). Run:
  `julia --project=benchmarks/ssm/bench_single_update -t 1 benchmarks/ssm/bench_single_update/bench_single_update.jl`
- `run_grid.sh`: single entry point running all 3 frameworks over a T/N
  sweep and aggregating results.
- `parse_results.py`: helper used by `run_grid.sh` to turn `RESULT,...` lines
  into a tidy long-format CSV.

## Quick start

```bash
brew install hyperfine   # used by libbi's bench-filter mode
./benchmarks/ssm/run_grid.sh
```

Produces `benchmarks/ssm/results/grid_results.csv` (tidy long format:
`framework,T,N,metric,value`) and `results/raw_results.log` (full raw output
per run).

## Sweep configuration

Two 1D sweeps (not a full T x N grid): vary N at a fixed T, and vary T at a
fixed N. All overridable via environment variables:

```bash
T_FIXED=5000 N_SWEEP="1000 10000 100000" \
N_FIXED=10000 T_SWEEP="1000 5000 20000" \
SEED=42 REPEATS=10 WARMUP=1 \
./benchmarks/ssm/run_grid.sh
```

`REPEATS`/`WARMUP` are passed through to libbi's `hyperfine`-based
`MODE=bench-filter` runner. The Julia scripts each do a single timed run per
grid point (already `@timed`, JIT-excluded via a warmup call) — increase
`REPEATS` for libbi only if you want tighter statistics there; re-run the
whole script multiple times if you want repeats for the Julia frameworks too.

## Timing methodology

- **WeightedSampling / SequentialMonteCarlo.jl**: `@timed run!(...)` /
  `@timed smc!(...)`, JIT compile time excluded via a small warmup call
  first (`stats.time - stats.compile_time`). Single-threaded process, so
  wall time ≈ CPU time (no OS-scheduling multi-core noise).
- **libbi**: `hyperfine` drives the repeated `libbi filter ...` runs
  (`--warmup`, `--runs`, `--export-json`). hyperfine's JSON export reports
  both wall-clock (mean/stddev/median/min/max) AND mean user/system CPU time
  per run (via `getrusage` on the child process) — no separate
  `/usr/bin/time` wrapping needed. Compare libbi's `wall_mean_s`/`user_mean_s`
  directly against the Julia frameworks' `elapsed_s` (both are
  single-process, single-threaded wall-clock measurements).
