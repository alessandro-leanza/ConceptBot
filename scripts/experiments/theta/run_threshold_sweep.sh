#!/usr/bin/env bash
set -euo pipefail

THETAS="${THETAS:-0.65 0.70 0.75 0.80 0.85}"
CATEGORIES="${CATEGORIES:-explicit_unambiguous}"
OUT_BASE="${OUT_BASE:-scripts/experiments/theta/results/threshold_sweep}"
NUM_TRIALS="${NUM_TRIALS:-1}"
PLANNER="${PLANNER:-direct}"
PLANNER_MODEL="${PLANNER_MODEL:-gpt-4o-mini}"
PREFETCH="${PREFETCH:-1}"
PRECOMPUTE="${PRECOMPUTE:-1}"
CACHE_ONLY="${CACHE_ONLY:-1}"
SAVE_POLICIES="${SAVE_POLICIES:-1}"
PLOT="${PLOT:-1}"

mkdir -p "$(dirname "${OUT_BASE}")"

python3 scripts/experiments/theta/validate_instructions.py

if [[ "${PREFETCH}" == "1" ]]; then
  PYTHONPATH=. python3 scripts/experiments/theta/prefetch_conceptnet_cache.py --categories ${CATEGORIES}
fi

if [[ "${PRECOMPUTE}" == "1" ]]; then
  PYTHONPATH=. python3 scripts/experiments/theta/precompute_similarity_cache.py --categories ${CATEGORIES}
fi

SWEEP_ARGS=(
  --theta-list ${THETAS}
  --categories ${CATEGORIES}
  --out "${OUT_BASE}"
  --num-trials "${NUM_TRIALS}"
  --planner "${PLANNER}"
  --planner-model "${PLANNER_MODEL}"
)

if [[ "${CACHE_ONLY}" == "1" ]]; then
  SWEEP_ARGS+=(--cache-only)
fi

if [[ "${SAVE_POLICIES}" == "1" ]]; then
  SWEEP_ARGS+=(--save-policies)
fi

if [[ "${PLOT}" == "1" ]]; then
  SWEEP_ARGS+=(--plot)
fi

PYTHONPATH=. python3 scripts/experiments/theta/threshold_sweep.py "${SWEEP_ARGS[@]}"
