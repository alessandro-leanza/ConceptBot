#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/theta/refresh_threshold_from_txt.sh <results.txt> [options]

Options:
  --category <name>     Override category instead of inferring it from the first line.
  --out <base>          Output base path. Default: <results.txt without .txt>
  --model <name>        Judge model. Default: gpt-4o-mini
  --batch-size <n>      Judge batch size. Default: 8
  --keep-judged         Keep intermediate judged JSON/CSV as <base>.judged.{json,csv}
  --no-plot             Skip PNG generation.
  -h, --help            Show this help message.
EOF
}

TXT_PATH=""
CATEGORY=""
OUT_BASE=""
MODEL="${MODEL:-gpt-4o-mini}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NO_PLOT=0
KEEP_JUDGED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --category)
      CATEGORY="${2:-}"
      shift 2
      ;;
    --out)
      OUT_BASE="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --no-plot)
      NO_PLOT=1
      shift
      ;;
    --keep-judged)
      KEEP_JUDGED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${TXT_PATH}" ]]; then
        echo "Only one results.txt path is supported." >&2
        usage >&2
        exit 1
      fi
      TXT_PATH="$1"
      shift
      ;;
  esac
done

if [[ -z "${TXT_PATH}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${TXT_PATH}" ]]; then
  echo "Results file not found: ${TXT_PATH}" >&2
  exit 1
fi

if [[ -z "${CATEGORY}" ]]; then
  CATEGORY="$(sed -n '1s/.*category=\([^ ]*\).*/\1/p' "${TXT_PATH}")"
fi

if [[ -z "${CATEGORY}" ]]; then
  echo "Could not infer category from ${TXT_PATH}. Use --category." >&2
  exit 1
fi

if [[ -z "${OUT_BASE}" ]]; then
  OUT_BASE="${TXT_PATH%.txt}"
fi

echo "[refresh] txt=${TXT_PATH}"
echo "[refresh] category=${CATEGORY}"
echo "[refresh] out_base=${OUT_BASE}"

TMP_JUDGED_BASE="${OUT_BASE}.judged_tmp"
ORIGINAL_JSON="${OUT_BASE}.json"
ORIGINAL_JSON_BAK="${OUT_BASE}.sweep_backup.json"

if [[ -f "${ORIGINAL_JSON}" && ! -f "${ORIGINAL_JSON_BAK}" ]]; then
  cp "${ORIGINAL_JSON}" "${ORIGINAL_JSON_BAK}"
  echo "[refresh] saved sweep backup ${ORIGINAL_JSON_BAK}"
fi

PYTHONPATH=. python3 scripts/experiments/theta/judge_threshold_results.py \
  "${TXT_PATH}" \
  --category "${CATEGORY}" \
  --out "${TMP_JUDGED_BASE}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}"

MERGE_SOURCE=""
if [[ -f "${ORIGINAL_JSON_BAK}" ]]; then
  MERGE_SOURCE="${ORIGINAL_JSON_BAK}"
elif [[ -f "${ORIGINAL_JSON}" ]]; then
  MERGE_SOURCE="${ORIGINAL_JSON}"
fi

if [[ -n "${MERGE_SOURCE}" ]]; then
  PYTHONPATH=. python3 scripts/experiments/theta/update_threshold_outputs_from_judged.py \
    "${MERGE_SOURCE}" \
    "${TMP_JUDGED_BASE}.json" \
    --out-json "${OUT_BASE}.json" \
    --out-csv "${OUT_BASE}.csv"

  if [[ "${NO_PLOT}" != "1" ]]; then
    if ! PYTHONPATH=. python3 scripts/experiments/theta/plot_threshold_results.py \
      "${OUT_BASE}.json" \
      --out "${OUT_BASE}.png"; then
      echo "[refresh] warning: original 3-subplot PNG generation failed." >&2
    fi
  fi
else
  if [[ "${KEEP_JUDGED}" == "1" ]]; then
    cp "${TMP_JUDGED_BASE}.json" "${OUT_BASE}.judged.json"
    cp "${TMP_JUDGED_BASE}.csv" "${OUT_BASE}.judged.csv"
  fi
  mv "${TMP_JUDGED_BASE}.json" "${OUT_BASE}.json"
  mv "${TMP_JUDGED_BASE}.csv" "${OUT_BASE}.csv"
  if [[ "${NO_PLOT}" != "1" ]]; then
    echo "[refresh] warning: no original sweep JSON available, so the 3-subplot PNG was not regenerated." >&2
  fi
fi

if [[ "${KEEP_JUDGED}" == "1" ]]; then
  if [[ -f "${TMP_JUDGED_BASE}.json" ]]; then
    mv "${TMP_JUDGED_BASE}.json" "${OUT_BASE}.judged.json"
  fi
  if [[ -f "${TMP_JUDGED_BASE}.csv" ]]; then
    mv "${TMP_JUDGED_BASE}.csv" "${OUT_BASE}.judged.csv"
  fi
fi

if [[ -f "${TMP_JUDGED_BASE}.json" || -f "${TMP_JUDGED_BASE}.csv" ]]; then
  rm -f "${TMP_JUDGED_BASE}.json" "${TMP_JUDGED_BASE}.csv"
fi

echo "[refresh] wrote ${OUT_BASE}.json"
echo "[refresh] wrote ${OUT_BASE}.csv"
if [[ "${NO_PLOT}" != "1" ]]; then
  echo "[refresh] wrote ${OUT_BASE}.png"
fi
