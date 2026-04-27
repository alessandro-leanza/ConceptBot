# Dynamic Property Induction Usage

This repository now includes a minimal, opt-in Dynamic Property Induction extension for generic OPE.

## Why This Was Added

The reviewer noted that OPE uses a manually predefined target-property list. Generic OPE checks properties such as `fragile`, `hold liquid`, and `dangerous`, while material OPE replaces that list with material labels such as `glass`, `plastic`, and `metal`.

That fixed attribute space can miss task-specific properties. For example, the instruction:

```text
Heat my food in the microwave.
```

may require properties such as:

- `microwave-safe`
- `heat-resistant`
- `non-metallic`
- `container`

These are not explicit targets in the default generic OPE property list.

## What Was Implemented

The implementation is intentionally small and backward-compatible.

New module:

```text
scripts/modules/dynamic_properties.py
```

Main functions:

- `induce_task_properties(...)`
- `normalize_property_name(...)`
- `merge_properties(...)`

Generic OPE in `scripts/modules/ope.py` now accepts optional parameters:

```python
dynamic_properties=None
dynamic_property_metadata=None
```

When `dynamic_properties` is `None`, OPE behavior is unchanged.

When `dynamic_properties` is provided, generic OPE merges those properties with the existing base properties and uses the merged list for ConceptNet relation filtering. The OPE prompt also asks the LLM to output `Yes/No` values for the additional task-relevant properties.

No integration was added yet for:

- `OPE_mat`
- `OPE_score`
- `OPE_score_par`
- `URP`
- `URP_risk`
- `threshold_sweep.py`

Those variants can be extended later using the same helper.

## How It Works

Dynamic property induction uses the existing OpenAI client helper from:

```text
scripts/modules/semantic_cache.py
```

It sends the user instruction, detected objects, optional task category, and base property list to the LLM. The request asks for JSON only.

The returned dictionary has this shape:

```python
{
    "dynamic_properties": [
        {
            "name": "microwave-safe",
            "description": "...",
            "reason": "...",
            "expected_value_type": "yes_no",
            "priority": 1,
        }
    ],
    "merged_properties": [
        "dangerous",
        "fragile",
        "deformable",
        "hold liquid",
        "safe",
        "stable",
        "poisonous",
        "microwave-safe",
    ],
    "rejected_properties": [],
    "source": "llm",
}
```

If the LLM call fails or returns invalid JSON, the function safely falls back to:

```python
{
    "dynamic_properties": [],
    "merged_properties": base_properties,
    "rejected_properties": [...],
    "source": "fallback",
}
```

## Run the Demo

From the repository root:

```bash
PYTHONPATH=. python3 scripts/experiments/demo_dynamic_properties.py
```

The demo uses:

```text
Instruction: Heat my food in the microwave.
Objects: aluminum tray, glass container, soup bowl, microwave oven
```

It prints:

- base properties
- induced dynamic properties
- merged properties

To also pass the induced properties into generic OPE:

```bash
PYTHONPATH=. python3 scripts/experiments/demo_dynamic_properties.py --run-ope
```

The `--run-ope` mode may call ConceptNet and the LLM for OPE reasoning, so it requires the same API/network setup as the rest of the ConceptBot pipeline.

## Why It Is Disabled by Default

The extension is opt-in because existing experiments and outputs should remain reproducible. The default OPE call signature still works, and dynamic properties are only used when explicitly passed to `OPE(...)`.

This preserves:

- existing OPE behavior,
- existing theta sweep behavior,
- existing cache behavior for `kind="ope_standard"`,
- existing URP and planner behavior.

When dynamic properties are enabled, generic OPE uses a separate cache kind:

```text
ope_standard_dynamic
```

## Limitations

- Dynamic properties depend on an LLM call and may be noisy.
- The LLM can propose properties that are useful-sounding but hard to verify with ConceptNet.
- Properties are capped and deduplicated, but the current duplicate map is intentionally simple.
- The generic OPE parser still stores values as strings.
- Material, risk, URP, and theta-sweep integration are not implemented yet.
- The demo is qualitative; it is not a benchmark.

## Future Extension Points

Later low-risk extensions:

- Pass dynamic properties into `OPE_mat` as compatibility evidence.
- Pass dynamic properties into `OPE_score_par` as risk evidence while preserving `Dangerous` and `DangerousWith` outputs.
- Surface dynamic property metadata in URP prompts.
- Add `--dynamic-property-induction` and `--max-dynamic-properties` to `threshold_sweep.py`.
- Add a small compatibility-focused evaluation category for microwave safety, heat resistance, waterproofness, food safety, and chemical interaction.
