# Benchmark Instructions And Specification

This directory contains the machine-readable specification of the 47-item
ConceptBot benchmark: `scenes.json` (planner input) and `gold.json` (admissible
policies used for scoring). The two files share the same item identifiers.
Both currently contain 47 items under a top-level `items` array (`schema_version: "2.0"`).

| Category key | Count |
| --- | ---: |
| `explicit_unambiguous` | 7 |
| `explicit_ambiguous` | 10 |
| `implicit` | 10 |
| `risk_aware` | 8 |
| `materials` | 7 |
| `toxicity` | 5 |

## `scenes.json` — SceneSpec

Runtime-facing. One record per item, with no reference answer:

```json
{
  "id": "ra_06",
  "category": "risk_aware",
  "instruction": "...",
  "scene": {
    "objects": [],
    "pickable_objects": [],
    "destinations": [],
    "entity_states": {},
    "forbid_self_placement": true
  }
}
```

`objects` is the complete scene inventory; `pickable_objects` and
`destinations` are explicit subsets whose union covers it. An entity may hold
both roles.

## `gold.json` — GoldSpec

Evaluation-facing. Each item defines the *set* of admissible policies, not a
single reference answer, in one of two representations:

- `sequences`: a list of accepted action sequences. All expert-validated
  alternatives satisfying the request are listed, up to 24 for `im_06`.
- `rules`: an object-to-destination assignment, used by the `materials` and
  `toxicity` items and by `ra_07`.

A policy is correct when it is a member of the admissible set. The `semantics`
block makes the matching contract explicit for every item:

| Field | Meaning |
| --- | --- |
| `match` | `exact_multiset`, `exact_sequence`, or `exact_assignment` |
| `order` | `unordered`, or `total` when causal order is enforced |
| `required_move_count` | exact number of `pick_and_place` actions |
| `allow_extra_actions` | unrequested actions are incorrect even when harmless |
| `allow_duplicate_actions` | repeated actions are incorrect |
| `require_final_done` | exactly one final `done()` is required |
| `forbid_self_placement` | source and destination must differ |

Ordering is enforced only for `ra_03`, `ra_06`, and `ra_08`, where stacking or
required support placement imposes a causal dependency. All other items use
unordered matching. `evaluation_scope` and `annotation_note` are evaluator and
annotation metadata; they are not planner inputs.

## Evaluation Boundary

`scenes.json` is the only runtime-facing file. `gold.json` is used solely for
post-generation scoring and is never loaded by OPE, URP, the Planner, or any
baseline while a policy is generated.

## Loading the Split Benchmark

The legacy loader below does not support the split files. Load runtime-facing
records directly from the repository root, without loading scoring annotations:

```python
import json
from pathlib import Path

scenes = json.loads(Path("instructions/scenes.json").read_text())
items = scenes["items"]
assert len(items) == 47
item = items[0]
print(item["id"], item["instruction"], item["scene"])
```

Configure OPE/URP using the item's category and use its explicit pickable objects
and destinations for the Planner. In a separate, post-generation scoring step,
join saved generated policies to `gold.json` by `id`. Do not pass `gold`,
`semantics`, or reference actions to any generation module.

The paper evaluates 10 complete policy-generation repetitions per instruction;
the instruction is the statistical unit. The supplied files specify admissible
outcomes, but this checkout does not include a complete evaluation runner/scorer.
The loader and demo commands do not reproduce the paper's reported metrics.

## Loader Example

`examples.json` is a compact instruction file documenting the legacy
category-file schema used by the loader and demo utilities:

```bash
python instructions/load_instructions.py
python instructions/load_instructions.py examples
```

The no-argument command lists JSON filenames, not only compatible categories.
Use `examples` with this loader: `scenes` and `gold` have a different schema and
must not be supplied as its category argument. The two example records embed
`gold` for illustration only; keep those fields outside runtime planner inputs.
