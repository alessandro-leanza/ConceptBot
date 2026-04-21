# Experiments

This folder contains scripts for evaluation and analysis.

## Threshold Sweep
Runs a theta sensitivity analysis across task categories.

```bash
PYTHONPATH=. python scripts/experiments/theta/threshold_sweep.py \
  --theta-list 0.65 0.70 0.75 0.80 0.85 \
  --categories implicit \
  --out scripts/experiments/theta/results/threshold_sweep \
  --cache-only \
  --num-trials 1 \
  --planner direct \
  --save-policies
```

Outputs:
- `scripts/experiments/theta/results/threshold_sweep.json`
- `scripts/experiments/theta/results/threshold_sweep.csv`
- `scripts/experiments/theta/results/threshold_sweep.txt` (if `--save-policies`)

Logging:
- default terminal output is compact progress only: `theta / category / instruction id / trial`
- use `--verbose` to re-enable module-level prints from OPE/URP/backend

## Docker Workflow
Build the experiment image once:

```bash
docker compose -f docker-compose.experiments.yml build
```

Run the full reproducible pipeline inside the container:

```bash
docker compose -f docker-compose.experiments.yml run --rm conceptbot-exp \
  bash scripts/experiments/theta/run_threshold_sweep.sh
```

Run the same pipeline while saving a full log and keeping live output in the terminal:

```bash
mkdir -p scripts/experiments/theta/results
docker compose -f docker-compose.experiments.yml run --rm conceptbot-exp \
  bash scripts/experiments/theta/run_threshold_sweep.sh 2>&1 | tee scripts/experiments/theta/results/threshold_sweep_run.log
```

This workflow:
- validates the instruction files
- optionally prefetches ConceptNet relations
- precomputes keyword / embedding / similarity caches
- runs the theta sweep
- writes plots if `PLOT=1`

Useful overrides:

```bash
docker compose -f docker-compose.experiments.yml run --rm \
  -e PREFETCH=0 \
  -e PRECOMPUTE=0 \
  -e CACHE_ONLY=1 \
  -e NUM_TRIALS=1 \
  -e CATEGORIES=implicit \
  -e PLANNER=direct \
  -e OUT_BASE=scripts/experiments/theta/results/threshold_sweep_cached \
  conceptbot-exp bash scripts/experiments/theta/run_threshold_sweep.sh
```

Notes:
- `.env` is injected at runtime via `env_file`, not copied into the image
- `cache/` and `scripts/experiments/theta/results/` persist on the host because the repo is bind-mounted into `/workspace`
- for a warm-cache offline sweep, first run with `PREFETCH=1 PRECOMPUTE=1`, then rerun with `CACHE_ONLY=1`

## Prefetch ConceptNet Cache
Populate local cache for offline runs.

```bash
PYTHONPATH=. python scripts/experiments/theta/prefetch_conceptnet_cache.py
```

## Precompute Semantic Cache
Populate keyword, embedding, and similarity caches so the theta sweep only needs to filter cached similarities.

```bash
PYTHONPATH=. python scripts/experiments/theta/precompute_similarity_cache.py
```

Generated caches:
- `cache/conceptnet_cache.json`
- `cache/keyword_cache.json`
- `cache/embedding_cache.json`
- `cache/similarity_cache.json`

## Validate Instructions
Sanity check for instruction JSON files.

```bash
python scripts/experiments/theta/validate_instructions.py
```

## Judge Threshold Results
Evaluate generated plans against the gold policies using deterministic matching first and batched LLM judging only for residual cases.

```bash
PYTHONPATH=. python scripts/experiments/theta/judge_threshold_results.py \
  scripts/experiments/theta/results/threshold_sweep_implicit.txt \
  --category implicit \
  --out scripts/experiments/theta/results/threshold_sweep_implicit_judged
```

Outputs:
- `scripts/experiments/theta/results/threshold_sweep_implicit_judged.json`
- `scripts/experiments/theta/results/threshold_sweep_implicit_judged.csv`
