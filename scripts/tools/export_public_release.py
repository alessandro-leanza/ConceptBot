#!/usr/bin/env python3
"""Create a sanitized public ConceptBot tree from the private working repo.

The export is allowlist-based: files are copied only when explicitly listed here.
This keeps raw experiments, reviewer notes, caches, and external baselines private by
construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

PUBLIC_FILES = [
    ".env.example",
    ".gitignore.public",
    "Dockerfile.public",
    "LICENSE",
    "README_PUBLIC.md",
    "docker-compose.public.yml",
    "requirements.txt",
    "instructions/README.public.md",
    "instructions/__init__.py",
    "instructions/load_instructions.py",
    "scripts/__init__.py",
    "scripts/demo_public.py",
    "scripts/modules/README.public.md",
    "scripts/modules/__init__.py",
    "scripts/modules/conceptnet_backend.py",
    "scripts/modules/dynamic_properties.py",
    "scripts/modules/ope.py",
    "scripts/modules/ope_score_par.py",
    "scripts/modules/pipeline_config.py",
    "scripts/modules/pl_vote.py",
    "scripts/modules/risk_trigger.py",
    "scripts/modules/semantic_cache.py",
    "scripts/modules/urp.py",
]

PRIVATE_PATTERNS = [
    "/cache/",
    "/results/",
    "/external_baselines/",
    "/docs/Reviewer_",
    "/docs/EXTERNAL_BASELINES_",
    "/scripts/experiments/",
    ".jsonl",
    ".log",
    "__pycache__",
    ".pyc",
]

EXAMPLE_INSTRUCTIONS = {
    "category": "examples",
    "shared_objects": [
        "energy bar",
        "water bottle",
        "chips",
        "trash can",
        "coke",
        "apple",
        "table",
        "user",
    ],
    "items": [
        {
            "id": "ex_01",
            "instruction": "Put an energy bar and a water bottle on the table.",
            "objects": None,
            "gold": {
                "sequences": [
                    [
                        "robot.pick_and_place(energy bar, table)",
                        "robot.pick_and_place(water bottle, table)",
                        "done()",
                    ]
                ]
            },
        },
        {
            "id": "ex_02",
            "instruction": "Can you throw away the apple and bring me a coke?",
            "objects": None,
            "gold": {
                "sequences": [
                    [
                        "robot.pick_and_place(apple, trash can)",
                        "robot.pick_and_place(coke, user)",
                        "done()",
                    ]
                ]
            },
        },
    ],
    "policy": {"order_matters": False, "type": "sequence"},
}



PUBLIC_DOCKERFILE = """FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

CMD ["python", "instructions/load_instructions.py"]
"""

PUBLIC_COMPOSE = """services:
  conceptbot:
    build:
      context: .
      dockerfile: Dockerfile
    working_dir: /workspace
    env_file:
      - .env
    environment:
      PYTHONPATH: /workspace
    volumes:
      - .:/workspace
    command: python instructions/load_instructions.py
"""

PUBLIC_INSTRUCTIONS_README = """# Instructions Examples

The public release includes a small example instruction file, `examples.json`, to document the benchmark schema without releasing the full internal benchmark suite.

## Schema

Each instruction file is a JSON object with:

- `category`: category name.
- `policy`: category-level metadata.
  - `order_matters`: whether action order matters.
  - `type`: `sequence`, `rules`, or `mixed`.
- `shared_objects`: optional object list shared by all items.
- `items`: instruction records.
  - `id`: stable identifier.
  - `instruction`: natural-language user request.
  - `objects`: item-specific object list, or `null` when `shared_objects` is used.
  - `gold`: example target policy.

## Loader Example

```bash
python instructions/load_instructions.py
python instructions/load_instructions.py examples
```
"""

PUBLIC_MODULES_README = """# Public Modules

This directory contains the core public ConceptBot modules:

- `conceptnet_backend.py`: ConceptNet access and relation normalization.
- `semantic_cache.py`: local cache helpers for embeddings, keywords, and LLM calls.
- `pipeline_config.py`: category-specific OPE/URP configuration.
- `dynamic_properties.py`: optional task-conditioned property expansion utilities.
- `risk_trigger.py`: optional risk-aware mode trigger utility.
- `ope.py`: unified Object Properties Extraction module.
- `urp.py`: unified User Request Processing module.
- `pl_vote.py`: deterministic next-action planner used in the public release.
- `ope_score_par.py`: risk-index OPE backend used by `ope.py` in risk mode.
"""

PUBLIC_DEMO = """#!/usr/bin/env python3
\"\"\"Minimal public ConceptBot demo.\n\nRequires `OPENAI_API_KEY` for LLM-backed OPE/URP/planning calls.\n\"\"\"\n\nfrom scripts.modules.ope import OPE\nfrom scripts.modules.pipeline_config import get_category_pipeline\nfrom scripts.modules.pl_vote import VOTE\nfrom scripts.modules.urp import URP\n\n\ndef main() -> None:\n    instruction = \"Bring me a coke and throw away the apple.\"\n    found_objects = [\"coke\", \"apple\", \"trash can\", \"user\", \"table\"]\n    pick_targets = [\"coke\", \"apple\"]\n    place_targets = {\"trash can\": None, \"user\": None, \"table\": None}\n\n    pipeline = get_category_pipeline(\"explicit_unambiguous\")\n    objects_info = OPE(\n        found_objects,\n        found_objects,\n        theta=0.75,\n        mode=pipeline[\"ope_mode\"],\n        pipeline_config=pipeline,\n        user_request=instruction,\n    )\n    structured_request = URP(\n        instruction,\n        found_objects,\n        objects_info,\n        use_OPE=True,\n        rel_objects=found_objects,\n        theta=0.75,\n        mode=pipeline[\"urp_mode\"],\n        pipeline_config=pipeline,\n    )\n    plan = VOTE(\n        found_objects=found_objects,\n        PICK_TARGETS=pick_targets,\n        PLACE_TARGETS=place_targets,\n        query=structured_request,\n        voting_samples=1,\n        temperature=0,\n    )\n\n    print(\"Instruction:\", instruction)\n    print(\"Structured request:\", structured_request)\n    print(\"Plan:\", plan)\n\n\nif __name__ == \"__main__\":\n    main()\n"""

PUBLIC_GITIGNORE = """# Local secrets and generated artifacts\n.env\ncache/\nresults/\ndist/\n__pycache__/\n*.pyc\n*.log\n*.jsonl\n\n# Internal/private experiment material\nscripts/experiments/**/results/\nscripts/experiments/**/old_results/\nexternal_baselines/\ndocs/Reviewer_*.md\ndocs/*REVISION*.md\ndocs/*RESPONSE*.md\ndocs/*BASELINE*.md\n"""


def _copy_file(src_rel: str, output: Path) -> None:
    dst_rel = src_rel
    if src_rel == "README_PUBLIC.md":
        dst_rel = "README.md"
    elif src_rel == ".gitignore.public":
        dst_rel = ".gitignore"
    elif src_rel == "Dockerfile.public":
        dst_rel = "Dockerfile"
    elif src_rel == "docker-compose.public.yml":
        dst_rel = "docker-compose.yml"
    elif src_rel == "instructions/README.public.md":
        dst_rel = "instructions/README.md"
    elif src_rel == "scripts/modules/README.public.md":
        dst_rel = "scripts/modules/README.md"
    dst = output / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_rel == ".gitignore.public":
        dst.write_text(PUBLIC_GITIGNORE)
        return
    if src_rel == "Dockerfile.public":
        dst.write_text(PUBLIC_DOCKERFILE)
        return
    if src_rel == "docker-compose.public.yml":
        dst.write_text(PUBLIC_COMPOSE)
        return
    if src_rel == "instructions/README.public.md":
        dst.write_text(PUBLIC_INSTRUCTIONS_README)
        return
    if src_rel == "scripts/modules/README.public.md":
        dst.write_text(PUBLIC_MODULES_README)
        return
    if src_rel == "scripts/demo_public.py":
        dst.write_text(PUBLIC_DEMO)
        return

    src = ROOT / src_rel
    if not src.exists():
        raise FileNotFoundError(src_rel)
    shutil.copy2(src, dst)


def _write_examples(output: Path) -> None:
    path = output / "instructions" / "examples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(EXAMPLE_INSTRUCTIONS, indent=2) + "\n")


def _iter_files(output: Path) -> Iterable[Path]:
    for path in output.rglob("*"):
        if path.is_file():
            yield path


def _validate_public_tree(output: Path) -> None:
    violations = []
    for path in _iter_files(output):
        rel = "/" + path.relative_to(output).as_posix()
        if rel == "/.env":
            violations.append(rel)
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern in rel or rel.endswith(pattern):
                violations.append(rel)
    if violations:
        joined = "\n".join(sorted(violations))
        raise RuntimeError(f"Private-looking files found in public export:\n{joined}")


def export_public(output: Path, force: bool) -> None:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        # Allow dist/ under the repo, but guard against accidental root overwrite.
        try:
            output.relative_to(ROOT / "dist")
        except ValueError as exc:
            raise ValueError("Output must be outside the repo or under ./dist") from exc

    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for rel in PUBLIC_FILES:
        _copy_file(rel, output)
    _write_examples(output)
    _validate_public_tree(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/conceptbot-public", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    export_public(args.output, args.force)
    print(f"Public export written to {args.output}")


if __name__ == "__main__":
    main()
