#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Defaults kept small since both Gen models are much slower per
# particle-step than benchmarks/ssm's WeightedSampling/SequentialMonteCarlo.jl/
# libbi (which run T=5000,N=10_000 in ~1s): the naive (no-Unfold) model is
# O(T^2*N) so T must stay tiny; the Unfold model is O(T*N) but ~100x slower
# per step, so both T and N are scaled down accordingly. The Unfold model is
# run at two (T, N) settings to show both a head-to-head comparable point
# (matching the naive T) and a larger-T point demonstrating its linear
# scaling. All overridable:
#   NAIVE_T=30 NAIVE_N=200 UNFOLD_T=200 UNFOLD_N=500 UNFOLD_T2=2000 UNFOLD_N2=200 SINGLE_N=500 ./run_gen.sh
NAIVE_T="${NAIVE_T:-50}"
NAIVE_N="${NAIVE_N:-10000}"
UNFOLD_T="${UNFOLD_T:-50}"
UNFOLD_N="${UNFOLD_N:-10000}"
UNFOLD_T2="${UNFOLD_T2:-1000}"
UNFOLD_N2="${UNFOLD_N2:-10000}"
SINGLE_N="${SINGLE_N:-1000}"

RESULTS_DIR="$ROOT_DIR/results"
RAW_LOG="$RESULTS_DIR/raw_results.log"
CSV_OUT="$RESULTS_DIR/gen_results.csv"

mkdir -p "$RESULTS_DIR"
: > "$RAW_LOG"

echo "[gen] naive (no Unfold) T=$NAIVE_T N=$NAIVE_N"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" naive "$NAIVE_T" "$NAIVE_N" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

echo "[gen] unfold T=$UNFOLD_T N=$UNFOLD_N"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" unfold "$UNFOLD_T" "$UNFOLD_N" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

echo "[gen] unfold T=$UNFOLD_T2 N=$UNFOLD_N2"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" unfold "$UNFOLD_T2" "$UNFOLD_N2" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

echo "[gen] single-update N=$SINGLE_N"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" single "$SINGLE_N" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

# Reuse benchmarks/ssm/parse_results.py -- same RESULT,<framework>,key=val,...
# line format, so results merge cleanly with benchmarks/ssm's tidy CSV
# schema (framework,T,N,metric,value). The "bench_single_update" framework
# rows here use the same metric-name convention (gen_*) as
# benchmarks/ssm/bench_single_update.jl's (ws_*/smc_*), so concatenating
# both raw logs before parsing gives one combined comparison table.
python3 "$ROOT_DIR/../ssm/parse_results.py" "$RAW_LOG" "$CSV_OUT"

echo ""
echo "Results written to $CSV_OUT"
echo "Raw per-run output log: $RAW_LOG"
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$CSV_OUT"
else
  cat "$CSV_OUT"
fi
