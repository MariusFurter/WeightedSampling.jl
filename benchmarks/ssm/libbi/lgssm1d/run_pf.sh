#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_FILE="LGSSM1D.bi"
DATA_FILE="data/obs.nc"
RESULTS_FILE="results/filter_bootstrap.nc"

# You can override these from the environment, e.g.
#   NPARTICLES=5000 T=200 ./run_pf.sh
T="${T:-100}"
NPARTICLES="${NPARTICLES:-2000}"
DATA_SEED="${DATA_SEED:-42}"
FILTER_SEED="${FILTER_SEED:-43}"
# Default 1.0 forces resampling at every step (relative ESS is always <= 1),
# matching the "always resample" default used by the WeightedSampling and
# SequentialMonteCarlo.jl benchmarks in this repo for a fair comparison.
ESS_REL="${ESS_REL:-1.0}"
NTHREADS="${NTHREADS:-1}"
MODE="${MODE:-all}"

# Repeated benchmark controls
REPEATS="${REPEATS:-10}"
TARGET_SECONDS="${TARGET_SECONDS:-5}"
CALIBRATION_MIN_PARTICLES="${CALIBRATION_MIN_PARTICLES:-100}"
CALIBRATION_MAX_PARTICLES="${CALIBRATION_MAX_PARTICLES:-50000000}"
WARMUP="${WARMUP:-1}"

mkdir -p data results

usage() {
  cat <<'EOF'
Usage:
  ./run_pf.sh
  MODE=<all|data|filter|bench-filter> [VAR=VALUE ...] ./run_pf.sh

Description:
  Run a simple LibBi benchmark for the 1D linear-Gaussian state-space model.
  Configuration is done via environment variables.

Modes:
  MODE=all          Generate data, then run filtering. (default)
  MODE=data         Generate data only.
  MODE=filter       Run filtering only; generates data if missing.
  MODE=bench-filter Repeated filter-only timing with summary statistics.
  MODE=help         Show this help and exit.

Core variables:
  T                 End time / number of model steps. Default: 100
  NPARTICLES        Particle count for filter.
                    For MODE=bench-filter, set to integer or 'auto'.
                    Default: 2000
  DATA_SEED         Seed for synthetic data generation. Default: 42
  FILTER_SEED       Base seed for filtering. Default: 43
  ESS_REL           ESS threshold for resampling. Default: 0.5
  NTHREADS          Number of CPU threads for libbi. Default: 1 (kept at 1 to
                    match the single-threaded Julia benchmark for a fair
                    comparison).

Benchmark variables (MODE=bench-filter):
  REPEATS                   Number of measured runs. Default: 10
  WARMUP                    Number of warmup runs. Default: 1
  TARGET_SECONDS            Target wall time per run when NPARTICLES=auto. Default: 5
  CALIBRATION_MIN_PARTICLES Starting particles for auto-calibration. Default: 100
  CALIBRATION_MAX_PARTICLES Upper limit for auto-calibration. Default: 50000000

Examples:
  ./run_pf.sh
  MODE=data T=5000 DATA_SEED=42 ./run_pf.sh
  MODE=filter T=5000 NPARTICLES=10000 FILTER_SEED=7 ./run_pf.sh
  MODE=bench-filter T=5000 NPARTICLES=auto TARGET_SECONDS=5 REPEATS=10 WARMUP=1 ./run_pf.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

generate_data() {
  echo "[data] Generating synthetic observations (joint sample)..."
  libbi sample \
    --target joint \
    --model-file "$MODEL_FILE" \
    --nsamples 1 \
    --start-time 0 \
    --end-time "$T" \
    --noutputs "$T" \
    --seed "$DATA_SEED" \
    --output-file "$DATA_FILE"
}

run_filter() {
  local nparticles="$1"
  local filter_seed="$2"

  libbi filter \
    --filter bootstrap \
    --model-file "$MODEL_FILE" \
    --obs-file "$DATA_FILE" \
    --start-time 0 \
    --end-time "$T" \
    --noutputs 0 \
    --nparticles "$nparticles" \
    --ess-rel "$ESS_REL" \
    --resampler systematic \
    --nthreads "$NTHREADS" \
    --seed "$filter_seed" \
    --output-file "$RESULTS_FILE"
}

ensure_data_file() {
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "[data] Missing $DATA_FILE, generating it now."
    generate_data
  fi
}

parse_time_file() {
  local time_file="$1"
  awk '
    /^real[[:space:]]+[0-9.]+/ { real = $2 }
    /^user[[:space:]]+[0-9.]+/ { user = $2 }
    /^sys[[:space:]]+[0-9.]+/  { sys = $2 }
    END {
      if (real == "" || user == "" || sys == "") {
        exit 1
      }
      print real, user, sys
    }
  ' "$time_file"
}

time_filter_once() {
  local nparticles="$1"
  local run_index="$2"
  local seed="$((FILTER_SEED + run_index))"
  local run_log="/tmp/libbi_pf_run_${run_index}.log"
  local time_log="/tmp/libbi_pf_time_${run_index}.log"

  /usr/bin/time -p -o "$time_log" \
    libbi filter \
      --filter bootstrap \
      --model-file "$MODEL_FILE" \
      --obs-file "$DATA_FILE" \
      --start-time 0 \
      --end-time "$T" \
      --noutputs 0 \
      --nparticles "$nparticles" \
      --ess-rel "$ESS_REL" \
      --resampler systematic \
      --nthreads "$NTHREADS" \
      --seed "$seed" \
      --output-file "$RESULTS_FILE" \
    >"$run_log" 2>&1

  parse_time_file "$time_log"
}

calibrate_nparticles_for_target() {
  local n="$CALIBRATION_MIN_PARTICLES"
  local parsed real_s user_s sys_s

  echo "[bench] Calibrating NPARTICLES to target ~${TARGET_SECONDS}s per run..." >&2
  while true; do
    parsed="$(time_filter_once "$n" 0)"
    read -r real_s user_s sys_s <<<"$parsed"
    echo "[bench] calibration N=$n => real=${real_s}s" >&2

    if awk -v r="$real_s" -v tgt="$TARGET_SECONDS" 'BEGIN { exit !(r >= tgt) }'; then
      echo "$n"
      return
    fi

    if (( n >= CALIBRATION_MAX_PARTICLES )); then
      echo "$n"
      return
    fi

    n=$((n * 2))
    if (( n > CALIBRATION_MAX_PARTICLES )); then
      n="$CALIBRATION_MAX_PARTICLES"
    fi
  done
}

benchmark_filter_repeated() {
  ensure_data_file

  local n_target="$NPARTICLES"
  if [[ "$NPARTICLES" == "auto" ]]; then
    n_target="$(calibrate_nparticles_for_target)"
  fi

  echo "[bench] Running repeated filter benchmark via hyperfine"
  echo "[bench] NPARTICLES=$n_target, REPEATS=$REPEATS, WARMUP=$WARMUP"

  if ! command -v hyperfine >/dev/null 2>&1; then
    echo "[bench] ERROR: hyperfine not found on PATH. Install via 'brew install hyperfine'." >&2
    exit 1
  fi

  local hf_json="/tmp/libbi_pf_hyperfine_T${T}_N${n_target}.json"
  local filter_cmd
  filter_cmd="$(printf 'libbi filter --filter bootstrap --model-file %q --obs-file %q --start-time 0 --end-time %q --noutputs 0 --nparticles %q --ess-rel %q --resampler systematic --nthreads %q --seed %q --output-file %q' \
    "$MODEL_FILE" "$DATA_FILE" "$T" "$n_target" "$ESS_REL" "$NTHREADS" "$FILTER_SEED" "$RESULTS_FILE")"

  hyperfine \
    --warmup "$WARMUP" \
    --runs "$REPEATS" \
    --export-json "$hf_json" \
    "$filter_cmd"

  # hyperfine's JSON export already reports mean/stddev/median/min/max wall
  # time AND mean user/system CPU time (via getrusage on the child process)
  # per run, so no separate `/usr/bin/time` wrapping is needed here.
  python3 - "$hf_json" "$T" "$n_target" <<'PYEOF'
import json
import sys

hf_json, t, n = sys.argv[1], sys.argv[2], sys.argv[3]
with open(hf_json) as f:
    data = json.load(f)
r = data["results"][0]

print()
print(f"[bench] Summary (NPARTICLES={n})")
print(f"[bench] wall   mean={r['mean']:.4f}s stddev={r['stddev']:.4f}s "
      f"median={r['median']:.4f}s min={r['min']:.4f}s max={r['max']:.4f}s")
print(f"[bench] user   mean={r['user']:.4f}s")
print(f"[bench] system mean={r['system']:.4f}s")
print(f"RESULT,libbi,T={t},N={n},wall_mean_s={r['mean']:.6f},wall_median_s={r['median']:.6f},"
      f"user_mean_s={r['user']:.6f},sys_mean_s={r['system']:.6f}")
PYEOF

  echo ""
  echo "Done."
  echo "  observations: $DATA_FILE"
  echo "  filter output: $RESULTS_FILE"
  echo "  hyperfine json: $hf_json"
}

case "$MODE" in
  all)
    echo "[mode=all] Data generation + filtering"
    generate_data
    echo "[filter] Running bootstrap particle filter..."
    run_filter "$NPARTICLES" "$FILTER_SEED"
    echo
    echo "Done."
    echo "  observations: $DATA_FILE"
    echo "  filter output: $RESULTS_FILE"
    ;;
  data)
    echo "[mode=data] Data generation only"
    generate_data
    echo
    echo "Done."
    echo "  observations: $DATA_FILE"
    ;;
  filter)
    echo "[mode=filter] Filtering only"
    ensure_data_file
    run_filter "$NPARTICLES" "$FILTER_SEED"
    echo
    echo "Done."
    echo "  filter output: $RESULTS_FILE"
    ;;
  bench-filter)
    echo "[mode=bench-filter] Repeated filtering benchmark"
    benchmark_filter_repeated
    ;;
  help)
    usage
    ;;
  *)
    echo "Unknown MODE='$MODE'. Expected one of: all, data, filter, bench-filter, help" >&2
    echo "" >&2
    usage >&2
    exit 2
    ;;
esac