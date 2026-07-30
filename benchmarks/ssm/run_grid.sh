#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Three fixed (T,N) scenarios (not a sweep/grid) for WeightedSampling,
# SequentialMonteCarlo.jl and libbi:
#   default:       T=1_000    N=1_000
#   high particle: T=1_000    N=1_000_000
#   long time:     T=1_000_000 N=1_000
# All overridable from the environment, e.g.
#   DEFAULT_T=2000 DEFAULT_N=2000 ./run_grid.sh
DEFAULT_T="${DEFAULT_T:-1000}"
DEFAULT_N="${DEFAULT_N:-1000}"
HIGH_PARTICLE_T="${HIGH_PARTICLE_T:-1000}"
HIGH_PARTICLE_N="${HIGH_PARTICLE_N:-1000000}"
LONG_TIME_T="${LONG_TIME_T:-1000000}"
LONG_TIME_N="${LONG_TIME_N:-1000}"
SEED="${SEED:-42}"

# Passed through to libbi's run_pf.sh MODE=bench-filter (hyperfine warmup/runs).
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-1}"

RESULTS_DIR="$ROOT_DIR/results"
RAW_LOG="$RESULTS_DIR/raw_results.log"
CSV_OUT="$RESULTS_DIR/grid_results.csv"

mkdir -p "$RESULTS_DIR"
: > "$RAW_LOG"

# All 3 frameworks are forced to resample at every step for these runs:
# WeightedSampling defaults ess_perc_min=1.0, SequentialMonteCarlo.jl defaults
# ess_perc_min=1.0 (passed through as essThreshold=1.0), and libbi's run_pf.sh
# now defaults ESS_REL=1.0 — no extra flags needed here to force it.

run_point() {
  local point_t="$1" point_n="$2"
  echo ""
  echo "=== T=$point_t N=$point_n ==="

  echo "[grid] WeightedSampling T=$point_t N=$point_n"
  julia --project="$ROOT_DIR/WeightedSampling" -t 1 "$ROOT_DIR/WeightedSampling/lgssm1d.jl" \
    "$point_t" "$point_n" | tee -a "$RAW_LOG" | grep '^RESULT,' || true

  echo "[grid] SequentialMonteCarlo T=$point_t N=$point_n"
  julia --project="$ROOT_DIR/SequentialMonteCarlo" -t 1 "$ROOT_DIR/SequentialMonteCarlo/lGModel.jl" \
    "$point_t" "$point_n" | tee -a "$RAW_LOG" | grep '^RESULT,' || true

  echo "[grid] libbi T=$point_t N=$point_n"
  (
    cd "$ROOT_DIR/libbi/lgssm1d"
    MODE=bench-filter T="$point_t" NPARTICLES="$point_n" DATA_SEED="$SEED" \
      REPEATS="$REPEATS" WARMUP="$WARMUP" ./run_pf.sh
  ) | tee -a "$RAW_LOG" | grep '^RESULT,' || true
}

echo "=== default ==="
run_point "$DEFAULT_T" "$DEFAULT_N"
echo "=== high particle ==="
run_point "$HIGH_PARTICLE_T" "$HIGH_PARTICLE_N"
echo "=== long time ==="
run_point "$LONG_TIME_T" "$LONG_TIME_N"

# Single-update (marginal-cost, one mutate+observe+resample step) benchmarks.
# WeightedSampling/SequentialMonteCarlo.jl are benched (with allocations) at
# N=1000,10000,100000 via bench_single_update.jl (unchanged, as before).
# libbi is benched (time only -- no easy allocation count for a compiled
# binary) at N=1000 via run_pf.sh's MODE=bench-single-update.
echo ""
echo "=== single update ==="
echo "[grid] WeightedSampling/SequentialMonteCarlo.jl single-update (N=1000,10000,100000)"
julia --project="$ROOT_DIR/bench_single_update" -t 1 "$ROOT_DIR/bench_single_update/bench_single_update.jl" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

echo "[grid] libbi single-update N=1000"
(
  cd "$ROOT_DIR/libbi/lgssm1d"
  MODE=bench-single-update NPARTICLES=1000 DATA_SEED="$SEED" \
    REPEATS="$REPEATS" WARMUP="$WARMUP" ./run_pf.sh
) | tee -a "$RAW_LOG" | grep '^RESULT,' || true

python3 "$ROOT_DIR/parse_results.py" "$RAW_LOG" "$CSV_OUT"

echo ""
echo "Combined tidy-format results written to $CSV_OUT"
echo "Raw per-run output log: $RAW_LOG"
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$CSV_OUT"
else
  cat "$CSV_OUT"
fi
