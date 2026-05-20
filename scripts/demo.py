#!/usr/bin/env python3
"""Minimal ConceptBot demo.

Requires `OPENAI_API_KEY` for LLM-backed OPE/URP/planning calls.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modules.ope import OPE
from scripts.modules.pipeline_config import get_category_pipeline
from scripts.modules.pl_vote import VOTE
from scripts.modules.urp import URP


def main() -> None:
    instruction = "Bring me a coke and throw away the apple."
    found_objects = ["coke", "apple", "trash can", "user", "table"]
    pick_targets = ["coke", "apple"]
    place_targets = {"trash can": None, "user": None, "table": None}

    pipeline = get_category_pipeline("explicit_unambiguous")
    objects_info = OPE(
        found_objects,
        found_objects,
        theta=0.75,
        mode=pipeline["ope_mode"],
        pipeline_config=pipeline,
        user_request=instruction,
    )
    structured_request = URP(
        instruction,
        found_objects,
        objects_info,
        use_OPE=True,
        rel_objects=found_objects,
        theta=0.75,
        mode=pipeline["urp_mode"],
        pipeline_config=pipeline,
    )
    plan = VOTE(
        found_objects=found_objects,
        PICK_TARGETS=pick_targets,
        PLACE_TARGETS=place_targets,
        query=structured_request,
        voting_samples=1,
        temperature=0,
    )

    print("Instruction:", instruction)
    print("Structured request:", structured_request)
    print("Plan:", plan)


if __name__ == "__main__":
    main()
