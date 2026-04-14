# Experiments

This folder contains scripts for evaluation and analysis.

## Threshold Sweep
Runs a theta sensitivity analysis across task categories.

```bash
PYTHONPATH=. python scripts/experiments/threshold_sweep.py \
  --theta-list 0.65 0.70 0.75 0.80 0.85 \
  --categories explicit_unambiguous explicit_ambiguous implicit risk_aware materials toxicity \
  --out results/threshold_sweep \
  --cache-only \
  --num-trials 5 \
  --save-policies
```

Outputs:
- `results/threshold_sweep.json`
- `results/threshold_sweep.csv`
- `results/threshold_sweep.txt` (if `--save-policies`)

## Prefetch ConceptNet Cache
Populate local cache for offline runs.

```bash
PYTHONPATH=. python scripts/experiments/prefetch_conceptnet_cache.py
```

## Validate Instructions
Sanity check for instruction JSON files.

```bash
python scripts/experiments/validate_instructions.py
```
