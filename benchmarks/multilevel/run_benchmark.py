#!/usr/bin/env python3
"""
run_benchmark.py

Benchmarks the hierarchical regression model (see `WeightedSampling/model.jl`
/ `numpyro/model.py`) across a grid of (J = number of groups, n_obs =
observations per group), comparing NumPyro (NUTS) against WeightedSampling
(SMC).

For each (J, n_obs):
  1. Simulate ground truth data (`simulate.jl`), shared by both implementations.
  2. Run NumPyro NUTS once (fixed num_warmup/num_samples), used as the
     reference inference quality (RMSE of posterior mean alpha_j vs the true
     group intercepts).
  3. Calibrate WeightedSampling's particle count N: starting from
     `N_START`, double N until its RMSE is <= NumPyro's RMSE * TOLERANCE (or
     `N_MAX` is reached), so both implementations reach comparable inference
     quality. Report the timing at that calibrated N.

Usage: python run_benchmark.py
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MULTILEVEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MULTILEVEL_DIR, "data")
WS_PROJECT = os.path.join(MULTILEVEL_DIR, "WeightedSampling")
NUMPYRO_DIR = os.path.join(MULTILEVEL_DIR, "numpyro")
NUMPYRO_PYTHON = os.path.join(NUMPYRO_DIR, "venv", "bin", "python")

# Grid: number of groups and observations/group both grow together.
CONFIGS = [
    (8, 5),
    (20, 10),
    (50, 20),
    (100, 40),
]

NUM_WARMUP = 500
NUM_SAMPLES = 1000
SEED = 42

N_START = 1000
N_MAX = 256_000
TOLERANCE = 1.25  # WS RMSE must be <= numpyro RMSE * TOLERANCE to "match" quality


def parse_kv_output(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            result[k] = float(v)
        except ValueError:
            result[k] = v
    return result


def simulate(J, n_obs, data_dir, seed=SEED):
    subprocess.run(
        ["julia", f"--project={REPO_ROOT}", os.path.join(MULTILEVEL_DIR, "simulate.jl"),
         str(J), str(n_obs), str(seed), data_dir],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )


def run_numpyro(data_dir):
    proc = subprocess.run(
        [NUMPYRO_PYTHON, os.path.join(NUMPYRO_DIR, "run_numpyro.py"), data_dir,
         "--num_warmup", str(NUM_WARMUP), "--num_samples", str(NUM_SAMPLES)],
        cwd=NUMPYRO_DIR, check=True, capture_output=True, text=True,
    )
    return parse_kv_output(proc.stdout)


def run_ws(data_dir, N):
    proc = subprocess.run(
        ["julia", f"--project={WS_PROJECT}", os.path.join(WS_PROJECT, "run_ws.jl"),
         data_dir, str(N)],
        cwd=MULTILEVEL_DIR, check=True, capture_output=True, text=True,
    )
    return parse_kv_output(proc.stdout)


def calibrate_ws(data_dir, target_rmse):
    """Double N starting from N_START until WS's rmse_alpha <= target_rmse
    (or N_MAX is reached). Returns the last (N, result) tried."""
    N = N_START
    result = run_ws(data_dir, N)
    while result["rmse_alpha"] > target_rmse and N < N_MAX:
        N *= 2
        result = run_ws(data_dir, N)
    return N, result


def main():
    rows = []
    for J, n_obs in CONFIGS:
        data_dir = os.path.join(DATA_DIR, f"J{J}_n{n_obs}")
        print(f"== J={J}, n_obs={n_obs} (total obs={J * n_obs}) ==", file=sys.stderr)

        print("  simulating data...", file=sys.stderr)
        simulate(J, n_obs, data_dir)

        print("  running numpyro (NUTS)...", file=sys.stderr)
        numpyro_result = run_numpyro(data_dir)

        target_rmse = numpyro_result["rmse_alpha"] * TOLERANCE
        print(f"  calibrating WeightedSampling N (target rmse_alpha <= {target_rmse:.4f})...", file=sys.stderr)
        N, ws_result = calibrate_ws(data_dir, target_rmse)

        rows.append((J, n_obs, numpyro_result, N, ws_result))

    header = (
        f"{'J':>5} {'n_obs':>6} {'total_obs':>10} | "
        f"{'numpyro_t(s)':>13} {'numpyro_rmse':>13} | "
        f"{'WS_N':>8} {'WS_t(s)':>9} {'WS_rmse':>9}"
    )
    print(header)
    print("-" * len(header))
    for J, n_obs, numpyro_result, N, ws_result in rows:
        print(
            f"{J:>5} {n_obs:>6} {J * n_obs:>10} | "
            f"{numpyro_result['elapsed_time']:>13.3f} {numpyro_result['rmse_alpha']:>13.4f} | "
            f"{N:>8} {ws_result['elapsed_time']:>9.4f} {ws_result['rmse_alpha']:>9.4f}"
        )


if __name__ == "__main__":
    main()
