#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Two 1D sweeps (vary N at fixed T, vary T at fixed N) rather than a full
# T x N grid. All overridable from the environment, e.g.
#   N_SWEEP="1000 5000" T_SWEEP="1000 5000" REPEATS=3 ./run_grid.sh
T_FIXED="${T_FIXED:-5000}"
N_SWEEP="${N_SWEEP:-1000 10000 100000}"
N_FIXED="${N_FIXED:-10000}"
T_SWEEP="${T_SWEEP:-1000 5000 20000}"
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

# Track visited (T,N) points to avoid re-running the shared fixed point twice
# if it happens to coincide between the two sweeps. Plain space-separated
# list (not an associative array) for compatibility with macOS's default
# bash 3.2 (no `declare -A` support).
seen=""

for n in $N_SWEEP; do
  key="${T_FIXED}:${n}"
  case " $seen " in
    *" $key "*) ;;
    *)
      run_point "$T_FIXED" "$n"
      seen="$seen $key"
      ;;
  esac
done

for t in $T_SWEEP; do
  key="${t}:${N_FIXED}"
  case " $seen " in
    *" $key "*) ;;
    *)
      run_point "$t" "$N_FIXED"
      seen="$seen $key"
      ;;
  esac
done

python3 "$ROOT_DIR/parse_results.py" "$RAW_LOG" "$CSV_OUT"

echo ""
echo "Combined tidy-format results written to $CSV_OUT"
echo "Raw per-run output log: $RAW_LOG"
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$CSV_OUT"
else
  cat "$CSV_OUT"
fi
