# ConceptBot: Knowledge-Graph–Grounded Commonsense for Task Decomposition in LLM Robot Planning

ConceptBot is a modular LLM-based robot planning framework for resolving underspecified pick-and-place requests with commonsense grounding. The system combines knowledge-graph retrieval with language-model reasoning to infer object properties, rewrite user requests into robot-ready instructions, and generate feasible pick-and-place action sequences.

## Overview

ConceptBot is organized around three main stages:

- **Object Properties Extraction (OPE)**: enriches detected objects with commonsense properties retrieved from ConceptNet and interpreted with an LLM.
- **User Request Processing (URP)**: rewrites natural-language user requests into structured task descriptions for the planner.
- **Planner**: selects feasible pick-and-place actions from the available action set and terminates with `done()` once the request is satisfied.

The implementation also includes category-specific configuration for standard, material-aware, toxicity-aware, and risk-aware tasks.

## Setup

Create a `.env` file from `.env.example` and set `OPENAI_API_KEY` for LLM-backed modules.

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
- `scripts/modules/ope.py`: Object Properties Extraction.
- `scripts/modules/urp.py`: User Request Processing.
- `scripts/modules/pl_vote.py`: pick-and-place planner.
- `scripts/modules/pipeline_config.py`: category-specific OPE/URP configuration.
- `scripts/demo.py`: minimal OPE -> URP -> Planner demo.
- `instructions/examples.json`: compact example instruction file.
- `instructions/load_instructions.py`: instruction loader utility.

## Examples

List available example categories:

```bash
python instructions/load_instructions.py
```

Load the included examples:

```bash
python instructions/load_instructions.py examples
```

Run the end-to-end demo:

```bash
python scripts/demo.py
```

## Citation

If you use this code, please cite the ConceptBot paper.
