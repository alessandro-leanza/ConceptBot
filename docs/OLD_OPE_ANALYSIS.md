# OPE Analysis

This document describes the current Object Property Extraction (OPE) implementation in this repository. It is based on the actual code in `scripts/modules/` and related experiment helpers.

## Purpose in the Paper

In the ConceptBot paper, OPE grounds detected scene objects in commonsense knowledge. For each object, it retrieves knowledge graph relations, filters them for relevance, and asks an LLM to infer object properties that help later stages make safe and feasible pick-and-place decisions. The paper-level role is to convert raw object names into task-useful semantic properties such as fragility, danger, stability, material, and risk.

## Purpose in the Current Code

In the current code, OPE functions take scene object names, retrieve ConceptNet-style triples, filter triples by embedding similarity to fixed target property labels, build an LLM prompt, and parse the LLM response into an `objects_info` dictionary.

That dictionary is then passed to URP, risk-aware URP, property-aware planner variants, or real-robot execution helpers depending on which pipeline flags are enabled.

The shared ConceptNet backend lives in `scripts/modules/conceptnet_backend.py`. Semantic caching and embedding similarity helpers live in `scripts/modules/semantic_cache.py`.

## Main Files

- `scripts/modules/ope.py`: generic binary-property OPE.
- `scripts/modules/ope_score.py`: generic scored-property OPE with 1-3 scores.
- `scripts/modules/ope_mat.py`: material extraction OPE.
- `scripts/modules/ope_score_par.py`: risk-index OPE with parallel object processing and optional Wikipedia/OpenIE fallback.
- `scripts/modules/conceptnet_backend.py`: ConceptNet retrieval through a Hugging Face Space plus local JSON cache.
- `scripts/modules/semantic_cache.py`: OpenAI client, embedding cache, keyword cache, relation-similarity cache.
- `scripts/experiments/theta/precompute_similarity_cache.py`: precomputes OPE and URP semantic caches for experiment categories.
- `scripts/ConceptBot_Main.py`: manual toggle-based entry point that selects OPE variants.
- `scripts/experiments/theta/threshold_sweep.py`: evaluation harness that selects OPE variants by category.
- `scripts/modules/pl_toplog_prop.py`: planner variant that injects OPE output into the scene prompt.
- `scripts/modules/pick_and_place.py`: real-robot helper that uses OPE output to adjust gripper behavior.

## OPE Variants

### `OPE` in `scripts/modules/ope.py`

Purpose: infer binary object properties.

Inputs:

- `found_objects`: list of scene objects included in the LLM user message.
- `rel_objects`: list of objects used for ConceptNet relation retrieval.
- `theta=0.75`: cosine-similarity threshold for keeping relations.
- `stats=None`: optional dictionary updated with relation-count statistics.
- `llm_temperature=0`: OpenAI chat-completion temperature.

Target properties:

```text
dangerous, fragile, deformable, hold liquid, safe, stable, poisonous
```

Output:

- Dictionary shaped like `{object_name: {property_name: value}}`.
- Parsed values are strings, typically `Yes` or `No`, because `parse_gpt_response()` stores response values directly.

Behavior:

- `get_conceptnet_relations(object_name)` requests relations `MadeOf`, `UsedFor`, `IsA`, `HasProperty`, `CapableOf`, `PartOf`, and `RelatedTo`.
- `get_cached_ope_similarities(..., kind="ope_standard")` computes or retrieves cached similarities between relation triples and the fixed target-property labels.
- Relations with `similarity >= theta` are appended to the LLM system prompt.
- The LLM is asked to return one block per object with binary properties.

### `OPE_score` in `scripts/modules/ope_score.py`

Purpose: infer object properties as 1-3 scores.

Inputs:

- `found_objects`
- `rel_objects`
- `theta=0.75`
- `stats=None`
- `llm_temperature=0`

Target properties:

```text
fragile, toxic, dangerous, stable, deformable
```

Output:

- Dictionary shaped like `{object_name: {property_name: int_score}}`.
- The parser only keeps integer values in the range 1-3.

Behavior:

- Relation retrieval uses the same OPE relation set as generic OPE.
- Similarities use `kind="ope_score"`.
- Important current-code detail: `use_kg = False` in `ope_score.py`, so by default this variant does not retrieve or inject ConceptNet relations unless that module-level flag is changed.

### `OPE_mat` in `scripts/modules/ope_mat.py`

Purpose: infer object materials for material-sorting tasks.

Inputs:

- `found_objects`
- `rel_objects`
- `theta=0.75`
- `stats=None`
- `llm_temperature=0`

Target materials:

```text
metal, plastic, glass, wood, ceramic, fabric, wax
```

Output:

- Dictionary shaped like `{object_name: {"materials": "..."} }` in practice, because `parse_gpt_response()` stores the `Materials:` line as a string.
- The prompt asks for a list of applicable materials, but the parser does not normalize the list into Python list values.

Behavior:

- Retrieves the same OPE ConceptNet relation set as generic OPE.
- Similarities use `kind="ope_materials"`.
- Uses `theta` to keep only material-relevant ConceptNet relations.
- Emits verbose logs of raw and filtered relations.

### `OPE_score_par` in `scripts/modules/ope_score_par.py`

Purpose: infer a risk index for each object and interaction-specific risks with other scene objects.

Inputs:

- `found_objects`
- `rel_objects`
- `user_request`
- `theta=0.75`
- `stats=None`
- `llm_temperature=0`

Target properties:

```text
dangerous
```

Output:

- Dictionary shaped like:

```python
{
    "object_name": {
        "score": int_or_none,
        "dangerous_with": ["other object (score)", ...],
    }
}
```

Behavior:

- Builds an embedding for only the target property `dangerous`.
- Processes objects in parallel with `ThreadPoolExecutor`.
- For each object, `process_object()` retrieves ConceptNet relations when `use_kg = True`.
- Similarities use `kind="ope_risk"`.
- Relations with `similarity >= theta` are included in the final risk prompt.
- The LLM is asked to assign a `Dangerous` score from 1 to 5 and a `DangerousWith` list of interacting objects that increase danger.
- `parse_gpt_response_with_context()` parses `Dangerous:` into `score` and `DangerousWith:` into a list of strings.

Wikipedia/OpenIE fallback:

- The module contains `fetch_wikipedia_content()`, `extract_openie_triples()`, `filter_triples_by_similarity()`, and `cluster_and_deduplicate_relations()`.
- The fallback is controlled by module-level `use_wiki = False`, so it is disabled by default.
- If enabled, it tries to fetch Wikipedia content, extract triples with Stanford OpenIE, keyword-filter triples, optionally deduplicate with DBSCAN, and optionally embedding-filter by similarity.
- The fallback has practical limitations: `fetch_wikipedia_content()` has an empty User-Agent, disambiguation handling is interactive, and some paths assume Stanford OpenIE is available.

## ConceptNet Retrieval

The repository does not directly call a local ConceptNet dump. `scripts/modules/conceptnet_backend.py` wraps a Hugging Face Space:

```python
Client("cstr/conceptnet_normalized")
```

The public function is:

```python
get_conceptnet_relations(word, lang="en", relations=None, cache_only=None)
```

It normalizes the word, builds a cache key from language, word, and relation list, and stores results in:

```text
cache/conceptnet_cache.json
```

If `CONCEPTNET_CACHE_ONLY=1`, cache misses return an empty relation list instead of making a network call.

OPE variants define their own wrapper `get_conceptnet_relations(object_name)` and request this relation set:

```text
MadeOf, UsedFor, IsA, HasProperty, CapableOf, PartOf, RelatedTo
```

## Semantic Filtering and Thresholds

The active OPE implementations use `get_cached_ope_similarities()` from `scripts/modules/semantic_cache.py`. That helper:

- embeds target labels such as `dangerous` or `glass`,
- embeds relation triples as text like `"apple IsA fruit"`,
- computes max cosine similarity between each relation triple and the target labels,
- caches the resulting relation/similarity list in `cache/similarity_cache.json`.

Each OPE variant then applies:

```python
filtered_relations = [
    (relation, similarity)
    for relation, similarity in relation_scores
    if similarity >= theta
]
```

The default threshold is `theta=0.75`. The theta sweep experiment varies this value with defaults:

```text
0.65, 0.70, 0.75, 0.80, 0.85
```

There are older local functions named `compute_relation_embeddings()`, `cosine_similarity()`, and `filter_relations_by_similarity()` in the OPE modules. In the active OPE flows, filtering is usually done through the cached helper instead of those local filter functions.

## How OPE Outputs Flow Downstream

OPE output is stored as `objects_info`.

Main paths:

- `scripts/ConceptBot_Main.py` sets `objects_info` by selecting one OPE variant, then passes it into `URP`, `URP_risk`, `POSNEG`, or `TOPLOG_prop` depending on flags.
- `scripts/experiments/theta/threshold_sweep.py` selects OPE by category:
  - `materials` uses `OPE_mat`.
  - `risk_aware` uses `OPE_score_par` and `URP_risk`.
  - all other categories use generic `OPE`.
- `scripts/modules/urp.py` and `scripts/modules/urp_risk.py` inject `str(objects_info)` into the system prompt when `use_OPE=True`.
- `scripts/modules/pl_toplog_prop.py` appends `objects_info` to the scene description under a `Properties:` section.
- `scripts/modules/pick_and_place.py` checks `objects_info[object_name]` and adjusts gripper settings if `fragile == "Yes"` or `hold liquid == "Yes"`.

## Material and Risk Differences from Generic OPE

Material OPE differs from generic OPE by replacing abstract safety/physical-property targets with material labels. It is intended for tasks where the robot must sort or reason by material. It still uses ConceptNet, embedding similarity, theta filtering, and an LLM parser.

Risk OPE differs more substantially:

- It only targets `dangerous`.
- It uses a 1-5 risk score rather than binary or 1-3 labels.
- It considers pairwise interaction risk through `DangerousWith`.
- It includes `user_request` in the LLM user message.
- It processes objects in parallel.
- It contains an optional Wikipedia/OpenIE fallback path that generic OPE does not use.

## Limitations

- Target properties are hard-coded inside each OPE function.
- There is no dynamic property induction from the user instruction; the system must already know which property family to extract.
- OPE variants duplicate a lot of retrieval, filtering, prompt, and parsing logic.
- Generic `OPE` parser stores values as strings and does not validate `Yes/No`.
- `OPE_mat` asks for lists but stores `Materials` as a raw string.
- `OPE_score` has `use_kg = False` by default, so it may produce scores without retrieved relations.
- Several module-level flags (`use_kg`, `use_wiki`, `use_example`) control behavior globally rather than per call.
- OpenAI model names are hard-coded in the modules.
- Prompt output parsing is brittle and depends on exact labels such as `Object:` and `DangerousWith:`.
- The Wikipedia fallback is disabled, interactive on disambiguation, and dependent on optional Stanford OpenIE.
- `stats` updates inside `OPE_score_par` happen while objects are processed with `ThreadPoolExecutor`; the shared dictionary is mutated from parallel tasks without synchronization.
- Imported modules such as `requests`, `re`, or `wikipediaapi` are unused or partially used in several files.
- The active relation-filter path is cached in `semantic_cache.py`, while older local filtering helpers remain in the modules, which can make the code harder to reason about.

## Opportunities for Extension

### Dynamic Property Induction

The smallest-disruption insertion point is before OPE calls `get_cached_ope_similarities()`.

Current flow:

```text
fixed target labels -> ConceptNet relation similarities -> theta filtering -> OPE prompt
```

Dynamic flow:

```text
user instruction + objects -> induced target properties -> ConceptNet relation similarities -> theta filtering -> OPE prompt
```

Minimal implementation approach:

- Add a new helper that proposes target properties from the user request and object list.
- Keep the output format as a list of short property labels.
- Pass those labels into the existing `get_cached_ope_similarities()` helper as `targets`.
- Reuse the existing theta-filtering and prompt-building logic.
- Keep generic OPE, material OPE, and risk OPE available as fixed-property modes.

Likely insertion sites:

- `OPE(...)` in `scripts/modules/ope.py`, because this is the default OPE used by most categories.
- `scripts/experiments/theta/threshold_sweep.py`, before category-specific OPE selection, if dynamic property induction should be evaluated as an experiment setting.
- `scripts/experiments/theta/precompute_similarity_cache.py`, so dynamic targets can be cached before sweeps.

### Other Extensions

- Normalize all OPE outputs into a typed schema instead of raw strings.
- Move target-property definitions to a configuration object or category-specific registry.
- Add per-call options instead of module-level flags.
- Replace exact-string parsing with JSON response formats where possible.
- Share OPE prompt-building and relation-filtering utilities across variants.
- Add tests for relation filtering, parser behavior, and malformed LLM responses.
- Make Wikipedia fallback non-interactive and require a valid User-Agent if it is enabled.
- Add an OPE mode that returns both induced properties and evidence relations for each property.
