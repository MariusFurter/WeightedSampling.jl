# LibBi benchmark: 1D linear-Gaussian SSM

This directory contains a minimal LibBi benchmark model and runner for a
univariate linear-Gaussian, discrete-time state-space model.

Model in `LGSSM1D.bi`:

- `x_0 ~ N(0, x0_std)`
- `x_t = a * x_{t-1} + w_t`, `w_t ~ N(0, q)`
- `y_t ~ N(x_t, r)`

## Files

- `LGSSM1D.bi`: LibBi model.
- `run_pf.sh`: Generates synthetic observations, then runs a bootstrap particle
  filter.

## Modes

`run_pf.sh` supports multiple modes via `MODE=`:

- `MODE=all` (default): generate data, then run filter.
- `MODE=data`: generate data only.
- `MODE=filter`: run filter only (reuses existing `data/obs.nc`; generates it if missing).
- `MODE=bench-filter`: run repeated filter-only timing with summary stats.

## Prerequisites

- `libbi` (see repo memory notes for Homebrew quirks).
- [`hyperfine`](https://github.com/sharkdp/hyperfine) for `MODE=bench-filter` (`brew install hyperfine`).
  hyperfine reports wall-clock mean/stddev/median/min/max AND mean user/system
  CPU time per run (via `getrusage`), so no separate `/usr/bin/time` wrapping
  is needed for CPU-time comparisons against BenchmarkTools-based Julia benchmarks.

## Quick start

From this directory:

```bash
chmod +x run_pf.sh
./run_pf.sh
```

Outputs:

- `data/obs.nc`: synthetic observations (generated with `libbi sample --target joint`)
- `results/filter_bootstrap.nc`: particle filter output

## Useful overrides

```bash
T=200 NPARTICLES=5000 DATA_SEED=123 FILTER_SEED=456 ESS_REL=1.0 ./run_pf.sh
```

`ESS_REL` defaults to `1.0` (force resampling at every step, since relative
ESS is always <= 1), matching the always-resample default used by the
WeightedSampling and SequentialMonteCarlo.jl benchmarks in `benchmarks/ssm/`
for a fair comparison.

Filter-only run (no data generation in the timed path):

```bash
MODE=filter T=200 NPARTICLES=5000 ./run_pf.sh
```

Generate data once:

```bash
MODE=data T=200 DATA_SEED=123 ./run_pf.sh
```

Repeated filter-only benchmark targeting ~5s per run:

```bash
MODE=bench-filter T=200 NPARTICLES=auto TARGET_SECONDS=5 REPEATS=10 WARMUP=1 ./run_pf.sh
```

The `NPARTICLES=auto` mode calibrates by doubling particles from
`CALIBRATION_MIN_PARTICLES` until one run reaches at least `TARGET_SECONDS`.

## Lightweight timing sweep

```bash
for N in 500 1000 2000 5000 10000; do
  echo "NPARTICLES=$N"
  /usr/bin/time -l env NPARTICLES=$N T=200 ./run_pf.sh >/tmp/libbi_pf_${N}.log 2>&1
  grep -E "real|user|sys|maximum resident" /tmp/libbi_pf_${N}.log || true
done
```

Or use the built-in repeated benchmark mode (recommended):

```bash
MODE=bench-filter T=200 NPARTICLES=auto TARGET_SECONDS=5 REPEATS=10 WARMUP=1 ./run_pf.sh
```

Notes:

- The first run can include LibBi code generation/build overhead.
- For fair timing, repeat runs and ignore the first warm-up run.
- On this machine, `libbi help ...` may fail due a missing Perl `Pod::Find`
  dependency in the Homebrew package, but `libbi sample` and `libbi filter`
  still work.