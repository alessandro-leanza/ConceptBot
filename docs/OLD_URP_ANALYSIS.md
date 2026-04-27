# URP Analysis

This document describes the current User Request Processing (URP) implementation in this repository. It is based on the actual code in `scripts/modules/` and related experiment helpers.

## Purpose in the Paper

In the ConceptBot paper, URP converts a user's underspecified or ambiguous natural-language request into a robot-actionable instruction. It uses commonsense knowledge to resolve intent, infer relevant objects or destinations, and produce a request that the downstream planner can turn into pick-and-place actions.

## Purpose in the Current Code

In the current code, URP takes a user message, the scene object list, optional OPE output, and ConceptNet context. It builds an LLM system prompt containing robot constraints, few-shot examples, object/property context, and filtered ConceptNet relations. The LLM response is expected to contain a `Reasoning:` section and an `Answer:` section. URP returns only the parsed `Answer`.

There are two main variants:

- `URP` in `scripts/modules/urp.py`.
- `URP_risk` in `scripts/modules/urp_risk.py`.

Both variants share the same broad retrieval/filtering structure.

## Inputs and Outputs

### `URP(...)` in `scripts/modules/urp.py`

Signature:

```python
URP(user_message, found_objects, objects_info, use_OPE, rel_objects, theta=0.75, stats=None, llm_temperature=0)
```

Inputs:

- `user_message`: original natural-language instruction.
- `found_objects`: list of scene objects used in the prompt.
- `objects_info`: OPE output dictionary.
- `use_OPE`: whether to inject `objects_info` into the prompt.
- `rel_objects`: objects used for ConceptNet relation retrieval.
- `theta=0.75`: threshold for relation filtering.
- `stats=None`: optional dictionary updated with relation-count statistics.
- `llm_temperature=0`: OpenAI chat-completion temperature.

Output:

- A string called `answer`, parsed from the LLM response after `Answer:`.
- This string is intended to be a robot-understandable instruction for the planner, not necessarily direct API-form actions.

### `URP_risk(...)` in `scripts/modules/urp_risk.py`

Signature:

```python
URP_risk(user_message, found_objects, objects_info, use_OPE, rel_objects, theta=0.75, stats=None, llm_temperature=0)
```

Inputs and output are the same shape as `URP`, but the prompt interprets `objects_info` as risk-evaluation data with 1-5 danger scores and `DangerousWith` interaction risks.

## Keyword Extraction Flow

Both `URP` and `URP_risk` use:

```python
extract_keywords_llm(text, llm_temperature=0)
```

This calls `get_cached_keywords()` from `scripts/modules/semantic_cache.py`.

Current keyword prompt:

```text
Extract the most important keywords (max 2) from the following text (must be single words):
```

The model is hard-coded as `gpt-4o-mini`. The result is split by comma into a list. There is no schema enforcement or validation that the result contains only one-word keywords.

There is an older spaCy-based keyword extraction block in comments, but it is not active.

## ConceptNet Relation Retrieval Flow

Both URP variants define a wrapper:

```python
get_conceptnet_relations(keyword)
```

The requested ConceptNet relation set is:

```text
IsA, UsedFor, HasProperty, CapableOf, MannerOf
```

The wrapper delegates to `scripts/modules/conceptnet_backend.py`, which uses the Hugging Face Space `cstr/conceptnet_normalized` and caches results in `cache/conceptnet_cache.json`.

The relation retrieval happens in three places in each URP call:

1. Keyword-to-request relations:
   - Extract keywords from `user_message`.
   - Retrieve ConceptNet relations for each keyword.
   - Compare each relation to the full user request.
   - Append filtered relations under `Relevant Relations from ConceptNet`.

2. Object-to-keyword relations:
   - Extract keywords again from `user_message`.
   - Retrieve ConceptNet relations for each object in `rel_objects`.
   - Compare each object relation to the extracted keywords.
   - Append filtered relations under `Relevant Object Relations from ConceptNet (Object-Keyword Similarity)`.

3. Keyword-relation-to-request relations:
   - Reuse the extracted keywords from the object-query block.
   - Retrieve ConceptNet relations for each keyword again.
   - Compare each relation to the full user request.
   - Append filtered relations under `Relevant Keyword Relations from ConceptNet (Keyword-User Request Similarity)`.

The keyword extraction and keyword relation retrieval are duplicated within one URP call.

## Object-Keyword Relation Filtering

Object-keyword filtering uses:

```python
get_cached_urp_object_keyword_similarities(...)
```

This helper in `scripts/modules/semantic_cache.py`:

- embeds each extracted keyword,
- embeds each relation triple as text,
- computes the maximum cosine similarity between the relation embedding and keyword embeddings,
- caches the list of `(relation, similarity)` values in `cache/similarity_cache.json`.

The URP module then keeps only:

```python
similarity >= theta
```

For keyword-to-request filtering, URP uses:

```python
get_cached_urp_request_similarities(...)
```

That helper embeds the full instruction and each relation triple, then computes cosine similarity between them.

There are local functions named `compute_relation_embeddings()`, `cosine_similarity()`, `filter_relations_by_similarity()`, and `filter_relations_by_similarity_obj_key()`. These reflect the same concept, but the active URP flow uses the cached helpers.

## Prompt Construction

Prompt construction is mostly inline in `URP` and `URP_risk`.

Common components:

- `system_goal`: role, robot constraints, safety guidance, and required response format.
- `system_examples`: few-shot natural-language examples.
- `system_env`: scene object list, or `objects_info` if `use_OPE=True`.
- ConceptNet relation sections appended after retrieval and filtering.

The LLM call:

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ],
    temperature=llm_temperature
)
```

Expected LLM response format:

```text
Reasoning:
...
Answer:
...
```

Parsing:

```python
reasoning = response_message.split("Reasoning:")[1].split("Answer:")[0].strip()
answer = response_message.split("Answer:")[1].strip()
```

Only `answer` is returned.

## Differences Between `URP` and `URP_risk`

The core retrieval and filtering logic is nearly identical. Differences are primarily in prompt wording and cache-key `kind` strings.

`URP`:

- Uses general object properties when `use_OPE=True`.
- Tells the model to consider stable and safe solutions.
- Mentions "properties" broadly.
- Uses cache kinds such as `urp_keyword_to_request`, `urp_object_to_keywords`, and `urp_keyword_relations_to_request`.

`URP_risk`:

- Treats `objects_info` as risk-evaluation output.
- Explains `Dangerous` score meanings from 1 to 5.
- Explains `DangerousWith` interaction risks from 1 to 5.
- Explicitly tells the model to avoid high `DangerousWith` scores 4 or 5 unless necessary.
- Uses cache kinds such as `urp_risk_keyword_to_request`, `urp_risk_object_to_keywords`, and `urp_risk_keyword_relations_to_request`.

The function signatures and return values are otherwise the same.

## Where OPE Outputs Are Used

OPE output enters URP through the `objects_info` argument when `use_OPE=True`.

In both URP variants:

```python
obj_info_str = str(objects_info)
system_env = "... " + obj_info_str + ...
```

The current implementation does not transform OPE output into a normalized prompt schema. It stringifies the Python dictionary and relies on the LLM to interpret it.

Pipeline locations:

- `scripts/ConceptBot_Main.py` calls `URP(...)` or `URP_risk(...)` after selecting an OPE variant.
- `scripts/experiments/theta/threshold_sweep.py` always sets `use_ope_in_urp = True` for its category-specific OPE outputs.

## Where Thresholds Are Used

Both URP variants expose `theta=0.75`.

Threshold use sites:

- Keyword relations are kept when relation-to-request similarity is `>= theta`.
- Object relations are kept when relation-to-keyword similarity is `>= theta`.
- Keyword relation retrieval in the second keyword block is also filtered by `>= theta`.

The same theta passed to OPE in the experiment harness is passed to URP, so OPE and URP relation retention move together in theta sweeps.

## Relationship to Planner

URP returns a natural-language answer. The planner receives this answer as the query.

Examples:

- `scripts/ConceptBot_Main.py` sets `urp_query` to `URP(...)`, `URP_risk(...)`, or the original user query, then passes `urp_query` into planner variants.
- `scripts/experiments/theta/threshold_sweep.py` stores the URP answer as `urp_action`, then passes it into `DIRECT(...)` or `ITER(...)`.
- If planner output is empty in the theta sweep, the harness tries to parse action-like content from the URP answer with `_extract_actions()`.

## Limitations

- Keyword extraction is LLM-based, hard-coded to `gpt-4o-mini`, and parsed by comma splitting.
- Keyword extraction is called twice in one URP execution when both KG request and KG object query paths are enabled.
- Relation retrieval for keyword relations can also be repeated in one call.
- Target relation types are hard-coded.
- The system prompt is assembled as a long string inside each function; there is no prompt template object or schema.
- `objects_info` is injected as `str(objects_info)` rather than a stable JSON or typed format.
- Response parsing is brittle. If the LLM omits `Reasoning:` or `Answer:`, the code will raise an index error.
- The code assumes natural-language instructions are enough for downstream planners; only the newer `DIRECT` planner enforces JSON actions.
- URP and `URP_risk` duplicate most implementation logic.
- Module-level flags (`use_request_processing`, `use_KG_query_objects`, `use_KG_query_request`, `use_example`) control behavior globally.
- `use_obj_query` is defined but not used in the active flow.
- There is no explicit handling for empty keyword lists. If keyword extraction returns an empty list, object-keyword similarity may fail because `max()` is called over an empty sequence in `get_cached_urp_object_keyword_similarities()`.
- Stats collection is partial. Keyword zero-relation counts are tracked for keyword-to-request retrieval, but object-query zero counts are not tracked in the same detailed way.
- There are no tests for malformed LLM responses, empty relation lists, empty keyword lists, or cache-only misses.

## Opportunities for Extension

### Dynamic Property Induction Integration

URP is a natural place to decide what properties matter for a request.

Minimal-disruption insertion point:

1. Run keyword extraction and possibly ConceptNet retrieval as URP already does.
2. Add a helper that derives task-relevant property labels from:
   - `user_message`,
   - `found_objects`,
   - retrieved keyword relations,
   - optionally current OPE outputs.
3. Pass those induced property labels to OPE before or alongside fixed-property extraction.
4. Inject the resulting dynamic properties into `system_env`.

This keeps URP's current role as the request-understanding stage while allowing OPE to become request-conditioned.

Candidate insertion locations:

- Before `URP(...)` is called in `scripts/ConceptBot_Main.py`, by adding a dynamic OPE step that receives the original user query.
- Inside `scripts/experiments/theta/threshold_sweep.py`, after loading `item["instruction"]` and before category-specific OPE selection, if the goal is evaluation.
- Inside `URP(...)`, before prompt construction, if dynamic property induction should be tightly coupled to request processing.

The lowest-risk architectural option is to implement dynamic property induction as a separate helper used by the pipeline before calling OPE. That avoids making URP responsible for both inducing properties and rewriting instructions.

### Other Extensions

- Deduplicate `URP` and `URP_risk` by extracting shared retrieval/filtering functions.
- Cache keyword extraction once per URP call and reuse it across all relation sections.
- Return a structured object containing `reasoning`, `answer`, used keywords, and retained relations.
- Use JSON response formats for URP output.
- Validate and normalize `objects_info` before injecting it into prompts.
- Add explicit fallback behavior for empty keywords and empty ConceptNet results.
- Make relation type lists configurable per task category.
- Add tests around prompt parsing and semantic filtering.
- Allow URP to request additional OPE properties when the user instruction mentions task-specific concepts not covered by fixed OPE targets.
