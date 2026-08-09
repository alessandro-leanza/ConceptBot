# Benchmark Instructions And Specification

This directory contains the machine-readable specification of the 47-item
ConceptBot benchmark: `scenes.json` (planner input) and `gold.json` (admissible
policies used for scoring). The two files share the same item identifiers.

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

## Loader Example

`examples.json` is a compact instruction file documenting the legacy
category-file schema used by the loader and demo utilities:

```bash
python instructions/load_instructions.py
python instructions/load_instructions.py examples
```
