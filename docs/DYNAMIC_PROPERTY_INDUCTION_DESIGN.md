# Dynamic Property Induction Design

This document proposes a lightweight extension for addressing the reviewer concern that ConceptBot's OPE module currently depends on manually predefined property targets such as `fragile`, `hold liquid`, `dangerous`, or material labels such as `glass`, `plastic`, and `metal`.

The goal is to optionally induce additional task-relevant properties from the user instruction, detected objects, and task category, then use those properties in the same ConceptNet retrieval/filtering and OPE reasoning path that already exists.

No code has been implemented yet.

## Reviewer Concern

The current OPE variants define their target attribute spaces manually:

- Generic OPE: `dangerous`, `fragile`, `deformable`, `hold liquid`, `safe`, `stable`, `poisonous`.
- Scored OPE: `fragile`, `toxic`, `dangerous`, `stable`, `deformable`.
- Material OPE: `metal`, `plastic`, `glass`, `wood`, `ceramic`, `fabric`, `wax`.
- Risk OPE: `dangerous`.

This can miss task-specific properties such as:

- `microwave-safe`
- `heat-resistant`
- `non-metallic`
- `waterproof`
- `food-safe`
- `chemically reactive`

The proposed extension should add these properties only when useful, without replacing the existing predefined targets.

## Recommended Insertion Point

Dynamic Property Induction should run before OPE relation-similarity filtering, then pass the merged target-property list into OPE.

Recommended flow:

```text
user instruction + detected objects + category
    -> induce dynamic properties
    -> merge with base OPE targets
    -> retrieve ConceptNet relations for each object
    -> filter relations against merged targets
    -> ask OPE LLM to infer base + dynamic properties
    -> pass enriched objects_info to URP / Planner
```

This is preferable to inserting it after OPE because the induced properties must influence which ConceptNet relations are retained. If induction happens after OPE, relation filtering has already discarded potentially relevant evidence.

This should be implemented as a shared utility used primarily by OPE and optionally surfaced to URP. It should not live only inside URP, because URP currently rewrites instructions after OPE has already run in the main pipeline and theta sweep.

## Proposed Module and Functions

Add a new module:

```text
scripts/modules/dynamic_properties.py
```

Primary function:

```python
def induce_task_properties(
    user_instruction: str,
    detected_objects: list[str],
    base_properties: list[str] | None = None,
    task_category: str | None = None,
    max_properties: int = 6,
    model: str = "gpt-4o-mini",
    llm_temperature: float = 0,
) -> dict:
    ...
```

Helper functions:

```python
def normalize_property_name(name: str) -> str:
    ...

def merge_properties(
    base_properties: list[str],
    dynamic_properties: list[str],
    max_dynamic_properties: int = 6,
) -> list[str]:
    ...
```

Optional cache helper:

```python
def get_cached_dynamic_properties(
    user_instruction: str,
    detected_objects: list[str],
    base_properties: list[str] | None = None,
    task_category: str | None = None,
    max_properties: int = 6,
    model: str = "gpt-4o-mini",
    llm_temperature: float = 0,
) -> dict:
    ...
```

The cache could be implemented later in `semantic_cache.py`, but the first implementation can keep the function uncached if the experiment size is small. For theta sweeps, caching is recommended to preserve reproducibility and cost control.

## Inputs

Exact input format:

```python
user_instruction = "Heat my food in the microwave."

detected_objects = [
    "aluminum tray",
    "glass container",
    "soup bowl",
    "microwave oven",
]

base_properties = [
    "dangerous",
    "fragile",
    "deformable",
    "hold liquid",
    "safe",
    "stable",
    "poisonous",
]

task_category = "implicit"  # or "materials", "risk_aware", "toxicity", None
max_properties = 6
```

`detected_objects` should be the same object list already passed into OPE as `found_objects` or `rel_objects`.

`base_properties` should be the fixed property list used by the selected OPE variant. If `None`, the function induces only dynamic properties and leaves merging to the caller.

`task_category` is optional and should be passed by the experiment harness when available. In `scripts/ConceptBot_Main.py`, it can remain `None` unless a manual category flag is added.

## Outputs

The primary function should return a dictionary:

```python
{
    "dynamic_properties": [
        {
            "name": "microwave-safe",
            "description": "Whether the object can safely be placed in a microwave oven.",
            "reason": "The user wants to heat food in a microwave; unsafe containers should be avoided.",
            "expected_value_type": "yes_no",
            "priority": 1
        },
        {
            "name": "heat-resistant",
            "description": "Whether the object can tolerate heating without melting, breaking, or releasing hazards.",
            "reason": "Heating requires containers that tolerate elevated temperature.",
            "expected_value_type": "yes_no",
            "priority": 2
        },
        {
            "name": "non-metallic",
            "description": "Whether the object is not made of metal.",
            "reason": "Metal containers are usually unsafe in microwaves.",
            "expected_value_type": "yes_no",
            "priority": 3
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
        "heat-resistant",
        "non-metallic"
    ],
    "rejected_properties": [
        {
            "name": "edible",
            "reason": "The property applies to food items but does not help choose a microwave-safe container in this scene."
        }
    ],
    "source": "llm"
}
```

The OPE callers should usually consume only:

```python
result["merged_properties"]
```

URP can optionally receive the richer `dynamic_properties` metadata for prompt context.

## Property Merging Rules

Dynamic properties must extend, not replace, base properties.

Rules:

1. Preserve base property order exactly for backward compatibility.
2. Normalize dynamic names to lowercase kebab-case or lowercase space-separated strings. Choose one convention and use it consistently. The least disruptive choice for current code is lowercase strings with spaces, because existing properties include `hold liquid`.
3. Drop dynamic properties that are exact duplicates of base properties after normalization.
4. Drop near-duplicates by a small synonym map:
   - `breakable` -> `fragile`
   - `toxic` -> `poisonous` or keep both only if the active variant already distinguishes them
   - `heat proof` -> `heat-resistant`
   - `microwave safe` -> `microwave-safe`
5. Cap the number of dynamic properties with `max_properties`.
6. Sort dynamic properties by LLM-provided priority before appending.
7. Never remove base properties.

Example:

```python
base_properties = ["dangerous", "fragile", "hold liquid"]
dynamic_properties = ["microwave-safe", "fragile", "heat-resistant"]
merged = ["dangerous", "fragile", "hold liquid", "microwave-safe", "heat-resistant"]
```

## Prompt Design

The induction prompt should ask for only properties useful for robot pick-and-place reasoning, not general object descriptions.

Recommended system prompt:

```text
You are helping a robot planning system decide which object properties should be checked before executing a pick-and-place task.

The robot can only pick objects and place them at destinations. Your job is to propose a small set of task-relevant object properties that would help decide which detected objects are suitable, safe, or risky for the user's instruction.

Rules:
- Return JSON only.
- Do not include properties that are already covered by the provided base_properties.
- Prefer concise property names that can be used as retrieval targets for commonsense knowledge.
- Use properties that can plausibly be inferred from object names and commonsense relations.
- Do not invent object-specific facts.
- Do not propose more than max_properties dynamic properties.
- Prefer safety, feasibility, compatibility, material, containment, and task-success properties.
- Avoid vague properties such as "useful", "appropriate", "good", or "relevant".
- Do not replace the base properties; only add missing task-relevant properties.
```

Recommended user prompt template:

```text
User instruction:
{user_instruction}

Detected objects:
{detected_objects_json}

Task category:
{task_category_or_null}

Base properties already checked:
{base_properties_json}

Maximum number of dynamic properties:
{max_properties}

Return exactly one JSON object with this schema:
{
  "dynamic_properties": [
    {
      "name": "short property name",
      "description": "one sentence explaining what the property means",
      "reason": "one sentence explaining why it matters for this task",
      "expected_value_type": "yes_no",
      "priority": 1
    }
  ],
  "rejected_properties": [
    {
      "name": "short property name",
      "reason": "why it was not needed"
    }
  ]
}
```

The OpenAI call should use:

```python
response_format={"type": "json_object"}
temperature=0
```

Expected example output:

```json
{
  "dynamic_properties": [
    {
      "name": "microwave-safe",
      "description": "Whether the object can safely be placed in a microwave oven.",
      "reason": "The instruction requires heating food in a microwave, so unsafe containers should be avoided.",
      "expected_value_type": "yes_no",
      "priority": 1
    },
    {
      "name": "heat-resistant",
      "description": "Whether the object can tolerate heating without melting, breaking, or releasing hazards.",
      "reason": "Objects used for heating should tolerate elevated temperature.",
      "expected_value_type": "yes_no",
      "priority": 2
    },
    {
      "name": "non-metallic",
      "description": "Whether the object is not made of metal.",
      "reason": "Metal objects are usually unsafe in microwave tasks.",
      "expected_value_type": "yes_no",
      "priority": 3
    },
    {
      "name": "container",
      "description": "Whether the object can contain food or liquid.",
      "reason": "The robot needs an object suitable for holding food while heating.",
      "expected_value_type": "yes_no",
      "priority": 4
    }
  ],
  "rejected_properties": [
    {
      "name": "edible",
      "reason": "The main decision is about safe heating containers, not whether the container itself is edible."
    }
  ]
}
```

## Integration Plan

### Generic OPE

Current generic OPE in `scripts/modules/ope.py` defines:

```python
properties = ['dangerous', 'fragile', 'deformable', 'hold liquid', 'safe', 'stable', 'poisonous']
```

Proposed optional signature extension:

```python
def OPE(
    found_objects,
    rel_objects,
    theta=0.75,
    stats=None,
    llm_temperature=0,
    dynamic_properties=None,
):
    ...
```

Minimal internal change:

```python
base_properties = [...]
properties = merge_properties(base_properties, dynamic_properties or [])
```

The merged property list should be used in:

```python
get_cached_ope_similarities(..., targets=properties, kind="ope_standard")
```

Prompt change:

- Keep the existing base-property format.
- Add a section for dynamic properties if present.
- Ask the model to output all base properties plus the dynamic properties.

Potential output concern:

- The existing parser can already parse arbitrary `Key: Value` lines into `objects_info`, so generic dynamic `Yes/No` properties can fit with minimal parser changes.
- The prompt should explicitly request `Yes/No` values for induced properties to match generic OPE behavior.

### Material OPE

Current material OPE defines:

```python
materials = ["metal", "plastic", "glass", "wood", "ceramic", "fabric", "wax"]
```

Dynamic induction should not replace these material labels. It should add task-specific compatibility properties when the task is not just classification by material.

Example for microwave:

```text
base material targets: metal, plastic, glass, wood, ceramic, fabric, wax
dynamic targets: microwave-safe, heat-resistant, non-metallic
```

Integration options:

1. Conservative: use dynamic properties only for relation filtering and prompt context, but keep output as `Materials`.
2. More useful: output both `Materials` and dynamic `Yes/No` properties.

Recommended first implementation:

- Add optional `dynamic_properties=None`.
- Use `materials + dynamic_properties` as similarity targets.
- Keep `Materials:` output unchanged.
- Add a separate prompt instruction: "Also consider the dynamic properties when deciding materials and suitability, but only output `Materials` in this variant."

This avoids breaking the current material parser. A later schema update can return both materials and dynamic compatibility labels.

### Risk OPE

Current risk OPE defines:

```python
properties = ["dangerous"]
```

Dynamic properties should be treated as evidence targets for retrieving better risk-relevant relations, not as new risk output fields in the first implementation.

Example:

```text
user instruction: "Put the cleaning chemicals near the food prep area."
dynamic targets: food-safe, chemically reactive, toxic, contaminating
```

Recommended first implementation:

- Add optional `dynamic_properties=None`.
- Use `["dangerous"] + dynamic_properties` as targets in `process_object()`.
- Keep output schema unchanged:
  - `Dangerous`
  - `DangerousWith`

Prompt change:

- Add a section saying dynamic properties are task-relevant risk evidence.
- Instruct the LLM to use them when assigning `Dangerous` and `DangerousWith`, but still output only the existing risk schema.

This preserves compatibility with `URP_risk`, which expects the current risk dictionary structure.

### URP

URP does not need to run induction itself in the minimal design. It should receive OPE outputs enriched by dynamic properties when generic OPE is used.

Optional URP integration:

- Pass `dynamic_property_metadata` into `URP` and `URP_risk`.
- Add a prompt section:

```text
Task-relevant dynamic properties considered by OPE:
- microwave-safe: Whether the object can safely be placed in a microwave oven.
- heat-resistant: Whether the object can tolerate heating.
```

Proposed optional signature extension:

```python
def URP(
    user_message,
    found_objects,
    objects_info,
    use_OPE,
    rel_objects,
    theta=0.75,
    stats=None,
    llm_temperature=0,
    dynamic_property_metadata=None,
):
    ...
```

This is useful but not required for the first implementation because `objects_info` is already injected into URP when `use_OPE=True`.

## Pipeline Integration

### `scripts/ConceptBot_Main.py`

Add a flag:

```python
dynamic_property_induction = False
```

When enabled:

```python
dynamic = induce_task_properties(
    user_instruction=user_query,
    detected_objects=found_objects_proc,
    base_properties=base_properties_for_selected_ope,
    task_category=None,
)
dynamic_properties = [p["name"] for p in dynamic["dynamic_properties"]]
```

Pass `dynamic_properties` into the selected OPE variant.

### `scripts/experiments/theta/threshold_sweep.py`

Add CLI flags:

```text
--dynamic-property-induction
--max-dynamic-properties 6
```

When enabled, call induction inside `_run_item_impl()` before category-specific OPE selection:

```python
dynamic = induce_task_properties(
    user_instruction=item["instruction"],
    detected_objects=found_objects,
    base_properties=base_properties_for_category,
    task_category=category,
    max_properties=args.max_dynamic_properties,
)
```

Then pass dynamic properties into `OPE`, `OPE_mat`, or `OPE_score_par`.

For reproducibility:

- Store induced properties in policy logs when `--save-policies` is enabled.
- Add aggregate columns such as `dynamic_properties_used` or write them to JSON results.

### `scripts/experiments/theta/precompute_similarity_cache.py`

If dynamic induction is used in theta sweeps, the precompute script should optionally induce properties per item and precompute similarities with the merged target list.

This is not required for a first qualitative case study, but it is needed for efficient repeated sweeps.

## Backward Compatibility

Default behavior must remain unchanged.

Requirements:

- `dynamic_property_induction=False` by default.
- Existing OPE function calls must still work.
- Existing target property lists must remain the default.
- Existing `kind` cache names should remain unchanged when dynamic induction is disabled.
- Existing theta sweep results must be reproducible when the flag is disabled.
- Existing parsers should not be forced to parse new schemas unless dynamic induction is enabled.
- Material and risk variants should preserve their current output structures in the first implementation.

Cache compatibility issue:

`get_cached_ope_similarities()` includes target labels in its cache key through the `anchor` and relation signature. Adding dynamic targets changes the cache key naturally. To make cache entries easier to inspect, use a distinct `kind`, for example:

```text
ope_standard_dynamic
ope_materials_dynamic
ope_risk_dynamic
```

Only use those dynamic cache kinds when induction is enabled.

## Evaluation Plan

The minimal evaluation should be a qualitative case study plus a small controlled instruction set.

### Qualitative Case Study 1: Microwave Compatibility

Instruction:

```text
Heat my food in the microwave.
```

Objects:

```text
aluminum tray, glass container, soup bowl, microwave oven, user
```

Expected dynamic properties:

```text
microwave-safe, heat-resistant, non-metallic, container
```

Expected benefit:

- The system should avoid the aluminum tray.
- It should prefer a microwave-safe glass or ceramic container if available.

### Qualitative Case Study 2: Heat Resistance

Instruction:

```text
Put the hot pan on something safe.
```

Objects:

```text
hot pan, plastic plate, wooden trivet, paper towel, table
```

Expected dynamic properties:

```text
heat-resistant, non-flammable, stable
```

Expected benefit:

- The system should prefer the trivet over plastic or paper.

### Qualitative Case Study 3: Waterproofness

Instruction:

```text
Move the item that can protect the phone from water to the user.
```

Objects:

```text
phone, paper bag, plastic pouch, cloth napkin, user
```

Expected dynamic properties:

```text
waterproof, protective, flexible
```

Expected benefit:

- The system should identify the plastic pouch rather than the paper bag or cloth napkin.

### Qualitative Case Study 4: Food Safety

Instruction:

```text
Put the snack in a safe container for food.
```

Objects:

```text
chips, chemical beaker, ceramic bowl, dusty box, user
```

Expected dynamic properties:

```text
food-safe, clean, container, non-toxic
```

Expected benefit:

- The system should prefer the ceramic bowl over the chemical beaker.

### Qualitative Case Study 5: Chemical or Toxic Interaction

Instruction:

```text
Move the cleaning chemical away from food items.
```

Objects:

```text
bleach bottle, apple, bread, counter, hazardous bin
```

Expected dynamic properties:

```text
toxic, contaminating, food-safe, chemically reactive
```

Expected benefit:

- The system should treat the chemical as risky near food and choose a safer destination.

### Metrics

For a lightweight evaluation:

- Compare baseline OPE vs dynamic OPE on the same scenes.
- Record induced properties.
- Record retained ConceptNet relations before and after induction.
- Record URP answer and planner actions.
- Manually judge whether the dynamic version captures task-relevant constraints missing in baseline.

For a slightly stronger evaluation:

- Add 10-20 new instruction items to a new category such as `compatibility`.
- Add gold policies.
- Run theta sweeps with dynamic induction disabled and enabled.
- Compare success rate and zero-relation counts.

## Risks and Caveats

- Induced properties may be noisy or too broad.
- The LLM may hallucinate properties that are not actually inferable from object names or ConceptNet.
- Dynamic properties may bias relation filtering toward irrelevant evidence.
- Too many dynamic properties can dilute filtering and increase prompt length.
- Some properties are negated concepts, such as `non-metallic`, and ConceptNet may not contain direct evidence for them.
- Dynamic properties should not replace predefined base properties.
- Dynamic properties should be capped, deduplicated, and normalized.
- Prompt output must be JSON and validated before use.
- If induction fails or returns invalid JSON, the pipeline should fall back to base properties.
- Evaluation should report induced properties so reviewers can inspect whether the extension is behaving sensibly.

## Recommended First Implementation Scope

Implement the smallest useful version:

1. Add `scripts/modules/dynamic_properties.py`.
2. Add `induce_task_properties()` with JSON output and validation.
3. Add optional `dynamic_properties=None` to generic `OPE`.
4. In generic `OPE`, merge dynamic properties with the existing property list and use the merged list for relation filtering and prompt output.
5. Add a manual flag in `scripts/ConceptBot_Main.py`.
6. Add a small qualitative script or documented example for microwave compatibility.

Defer until after the first working version:

- Full material/risk variant integration.
- URP signature changes.
- Theta sweep CLI flags.
- Dynamic-property cache precomputation.
- New benchmark categories.

This keeps the implementation low-risk while directly addressing the reviewer comment.
