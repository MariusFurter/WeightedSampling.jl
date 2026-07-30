#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Turing is only run at the suite-wide "default" scenario point (T=1000,
# N=1000, same as WeightedSampling/SequentialMonteCarlo.jl/libbi's default in
# run_grid.sh and GenParticleFilters' UNFOLD_T2/UNFOLD_N2 in run_gen.sh) since
# Turing's SMC is much slower per particle-step than those frameworks.
# Mirrors benchmarks/GenParticleFilters/run_gen.sh's structure. Overridable:
#   FULL_T=1000 FULL_N=1000 SINGLE_N=1000 ./run_turing.sh
FULL_T="${FULL_T:-1000}"
FULL_N="${FULL_N:-1000}"
SINGLE_N="${SINGLE_N:-1000}"

RESULTS_DIR="$ROOT_DIR/results"
RAW_LOG="$RESULTS_DIR/raw_results.log"
CSV_OUT="$RESULTS_DIR/turing_results.csv"

mkdir -p "$RESULTS_DIR"
: > "$RAW_LOG"

echo "[turing] full run T=$FULL_T N=$FULL_N"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" full "$FULL_T" "$FULL_N" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

echo "[turing] single-update N=$SINGLE_N"
julia --project="$ROOT_DIR" "$ROOT_DIR/lgssm1d.jl" single "$SINGLE_N" \
  | tee -a "$RAW_LOG" | grep '^RESULT,' || true

# Reuse benchmarks/ssm/parse_results.py -- same RESULT,<framework>,key=val,...
# line format, so results merge cleanly with benchmarks/ssm's/GenParticleFilters'
# tidy CSV schema (framework,T,N,metric,value). The "bench_single_update"
# framework rows here use a turing_-prefixed metric name (vs ws_/smc_/gen_),
# so concatenating raw logs before parsing gives one combined comparison table.
python3 "$ROOT_DIR/../ssm/parse_results.py" "$RAW_LOG" "$CSV_OUT"

echo ""
echo "Results written to $CSV_OUT"
echo "Raw per-run output log: $RAW_LOG"
if command -v column >/dev/null 2>&1; then
  column -s, -t < "$CSV_OUT"
else
  cat "$CSV_OUT"
fi
