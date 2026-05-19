# ConceptBot Public Release

ConceptBot is a modular LLM-based robot planning framework for resolving underspecified pick-and-place requests with commonsense grounding. The public release contains the core OPE, URP, and Planner modules plus a small example instruction file. Internal experiment traces, working notes, tuning runs, and vendored external baselines are intentionally excluded.

## Public Scope

Included:
- Core modules for ConceptNet access, object-property extraction, user-request processing, and planning.
- Minimal instruction examples for demonstrating the data schema.
- Docker and Python dependency files for running the core code.

Excluded:
- Full benchmark suites and gold policies.
- Raw experiment outputs, logs, JSONL traces, and plots.
- Working notes, draft analysis documents, and internal planning material.
- Vendored third-party baseline repositories and their datasets.

## Setup

Create a `.env` file from `.env.example` and set `OPENAI_API_KEY` if you run LLM-backed modules.

```bash
cp .env.example .env
```

Install dependencies locally:

```bash
pip install -r requirements.txt
```

Or build the Docker image:

```bash
docker compose build
```

## Repository Layout

- `scripts/modules/`: core ConceptBot modules.
- `scripts/demo_public.py`: minimal public demo for the OPE -> URP -> Planner pipeline.
- `instructions/examples.json`: small public example dataset showing the instruction schema.
- `instructions/load_instructions.py`: helper for loading instruction JSON files.

## Notes

This public release is meant for code inspection and lightweight reproduction of the pipeline structure. Full experimental campaigns and paper-internal benchmark traces are kept private to avoid releasing raw LLM outputs and intermediate tuning artifacts.
