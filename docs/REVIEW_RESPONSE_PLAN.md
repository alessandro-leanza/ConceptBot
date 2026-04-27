# Review Response Plan

This document maps the reviewer comments to the current repository changes and proposes the remaining work needed for a strong revision package.

## Current Changes That Already Help

### Dynamic Property Induction

Relevant files:

- `scripts/modules/dynamic_properties.py`
- `scripts/experiments/demo_dynamic_properties.py`
- `docs/DYNAMIC_PROPERTY_INDUCTION_DESIGN.md`
- `docs/DYNAMIC_PROPERTY_INDUCTION_USAGE.md`
- `scripts/modules/ope.py`

This directly addresses Reviewer 1's concern that OPE relies on a manually predefined attribute space. The new mechanism can optionally induce task-specific properties from the instruction and detected objects, for example:

- `microwave safe`
- `heat resistant`
- `non metallic`
- `container type`

This is a good response to the example in the review: heat resistance and microwave compatibility. However, the current implementation should be presented as a lightweight optional extension, not as a fully validated solution yet.

Remaining action:

- Add a short qualitative case study in the paper showing baseline OPE vs dynamic-property OPE.
- Report induced properties and show how they change the OPE prompt/reasoning.
- Be honest that induced properties are capped, validated, and used as additional OPE targets rather than replacing the predefined properties.

### Theta Sweep

Relevant files:

- `scripts/experiments/theta/threshold_sweep.py`
- `scripts/experiments/theta/results/`
- `scripts/experiments/theta/plot_combined_threshold_results.py`
- `scripts/experiments/theta/plot_combined_threshold_from_table.py`

This addresses Reviewer 1 and Reviewer 2's concern that the cosine similarity threshold `theta = 0.75` was empirically chosen. A threshold sweep can be used as the systematic sensitivity analysis requested by both reviewers.

Remaining action:

- Run or finalize sweeps across categories, not only one category.
- Report success rate or policy correctness as a function of theta.
- Discuss the precision-recall tradeoff:
  - low theta retains more ConceptNet relations but increases noise;
  - high theta filters noise but may discard useful long-tail relations;
  - `0.75` can be justified only if sweep results show it is robust or near-optimal.
- Include a plot/table in the paper.

### Category Pipeline Refactor

Relevant files:

- `scripts/modules/pipeline_config.py`
- `scripts/modules/ope.py`
- `scripts/modules/urp.py`
- `scripts/modules/README.md`

This does not directly answer a single reviewer comment by itself, but it makes the method easier to explain and defend. The paper can now describe category-specific OPE/URP behavior explicitly:

- standard categories use standard OPE and standard URP;
- `materials` uses material-specific properties and material-specific URP;
- `toxicity` uses toxicity-specific binary properties and toxicity-specific URP;
- `risk_aware` uses risk OPE and risk URP.

This helps with reviewer concerns about hidden manual choices because the category-specific properties and prompts are centralized and inspectable.

## Reviewer Comment Matrix

| Source | Comment | Already Addressed By | Still Needed |
| --- | --- | --- | --- |
| Editor 1 | Explain modifications and highlight changes in manuscript | Not a code issue | Prepare response letter with exact paper locations and highlighted manuscript changes |
| Editor 2 | Upload experimental video | Not addressed | Produce and upload mp4 video |
| Reviewer 1 | OPE uses predefined attributes and may miss heat resistance/microwave compatibility | Dynamic Property Induction | Add paper section and qualitative evaluation |
| Reviewer 1 | Fixed theta = 0.75 lacks generality | Theta sweep infrastructure | Complete sweep results and add sensitivity plot/table |
| Reviewer 1 | Risk Index invoked only when user explicitly requires enhanced safety | Category pipeline helps only when category is known | Add implicit risk trigger or discuss limitation |
| Reviewer 1 | Risk Index is semantic, not grounded physical reasoning | Not addressed in code | Add limitation/future work, optionally add failure cases |
| Reviewer 2.1 | Need theta sensitivity analysis | Theta sweep infrastructure | Report results in paper |
| Reviewer 2.2 | Too few baselines beyond SayCan | Not addressed | Add baseline or justify scope |
| Reviewer 2.3 | Need voting `n` analysis | Not addressed | Add planner voting ablation |
| Reviewer 2.4 | Need real-world failure cases | Not addressed | Add qualitative failure cases and discussion |
| Reviewer 2.5 | OPE/URP overhead is non-negligible | Not addressed | Add runtime measurements and optimization discussion |
| Reviewer 2.6 | No-KG ablation keeps CoT, possibly underestimating KG contribution | Not addressed | Add fairer ablation or discuss limitation |

## Detailed Response Strategy

### 1. Predefined OPE Attribute Space

Reviewer concern:

The OPE target attributes are manually specified. For example, generic OPE checks `fragile`, `hold liquid`, and `dangerous`, while material OPE checks labels such as `glass`, `plastic`, and `metal`. This can miss task-specific properties such as microwave compatibility.

Current response:

Dynamic Property Induction adds an optional step before OPE relation filtering:

```text
instruction + detected objects
    -> induce task-relevant properties
    -> merge with base OPE properties
    -> retrieve/filter ConceptNet relations
    -> run OPE prompt on base + dynamic properties
```

Why this addresses the comment:

- The attribute space is no longer only predefined when the option is enabled.
- The predefined base properties remain for reproducibility.
- Dynamic properties are task-conditioned and can include properties absent from the original property list.

Suggested paper update:

- Add a paragraph in the method section after OPE explaining optional Dynamic Property Induction.
- Add a small figure or text flow showing where it enters the pipeline.
- Add a qualitative example:

```text
Instruction: Heat my food in the microwave.
Objects: aluminum tray, glass container, soup bowl, microwave oven.
Induced properties: microwave safe, heat resistant, non metallic, container type.
```

Important caveat:

The current demo showed that the LLM may still infer wrong values when ConceptNet evidence is sparse, such as incorrectly labeling an aluminum tray as microwave safe. This should be treated as an honest limitation and motivation for better material grounding or stricter evidence use.

### 2. Cosine Similarity Threshold

Reviewer concern:

The paper uses a fixed `theta = 0.75`, but this value may not generalize across object categories or task contexts.

Current response:

The theta sweep infrastructure can systematically evaluate multiple thresholds.

Recommended experiment:

- Run threshold sweeps for:
  - `explicit_unambiguous`
  - `explicit_ambiguous`
  - `implicit`
  - `materials`
  - `toxicity`
  - `risk_aware`
- Use the same instruction set and seeds for each theta.
- Report:
  - policy success rate;
  - number of retained ConceptNet relations;
  - number of zero-relation objects;
  - failure cases at too-low and too-high theta.

Suggested manuscript claim:

Do not say that `0.75` is universally optimal. Say that it was selected based on the observed tradeoff in the sweep and that adaptive thresholds are future work.

Possible stronger extension:

Add an adaptive theta mode later:

```text
start at theta = 0.75
if no relations are retained, lower theta to 0.65
if too many relations are retained, keep top-k by similarity
```

This would directly address the long-tail object concern, but it is more code and should be separate from the current category-pipeline refactor.

### 3. Risk Index Activation

Reviewer concern:

The risk OPE is invoked only when the user explicitly requests safety. A task may be risky even if the user does not mention risk.

Current response:

The category pipeline makes risk behavior explicit for `risk_aware`, but it does not solve implicit risk detection by itself.

Recommended next change:

Add a lightweight risk-trigger classifier before selecting the pipeline:

```text
instruction + detected objects
    -> detect implicit risk cues
    -> if risk cues exist, enable risk OPE or add risk properties
```

Possible risk cues:

- object names: `knife`, `glass`, `bleach`, `chemical`, `hot pan`, `medicine`, `battery`, `needle`
- instruction keywords: `heat`, `cut`, `clean`, `dispose`, `drink`, `eat`, `child`, `food`, `sharp`, `hot`
- induced dynamic properties: `toxic`, `flammable`, `heat resistant`, `food safe`, `microwave safe`, `sharp`, `contaminating`

Minimal implementation idea:

- Add a `risk_trigger` utility that returns:

```python
{
    "risk_detected": True,
    "risk_reasons": ["object 'bleach bottle' suggests toxicity"],
    "recommended_mode": "risk"
}
```

- Keep it disabled by default for backward compatibility.
- In the paper, position this as future work unless implemented and evaluated.

### 4. Semantic Risk vs Grounded Physical Risk

Reviewer concern:

Risk Index relies on semantic knowledge, not physical simulation or grounded physical reasoning.

Current response:

Not addressed by current code.

Recommended paper response:

- Acknowledge this limitation explicitly.
- Clarify that ConceptBot estimates commonsense semantic risk, not physics-validated execution risk.
- Add examples of risks outside current scope:
  - unstable center of mass;
  - object slipping during grasp;
  - thermal deformation;
  - liquid sloshing;
  - unstable placement geometry.

Optional experiment:

Add qualitative failure cases from real-world runs where semantic planning was correct but physical execution failed. This also addresses Reviewer 2.4.

Future work wording:

ConceptBot could combine semantic risk with:

- force/torque feedback;
- object pose and support polygon estimates;
- physics simulation;
- affordance prediction;
- learned grasp stability scores.

### 5. Baselines Beyond SayCan

Reviewer concern:

The paper mainly compares against SayCan, which may be insufficient.

Current response:

Not addressed by current repository changes.

Options:

1. Add one or two more baselines if feasible.
2. If not feasible, strengthen the justification for why SayCan is the primary baseline and add an explicit limitation.

Possible baselines:

- LLM-only planner without ConceptNet.
- LLM + OPE but no URP.
- LLM + URP but no OPE.
- No-KG ConceptBot with matched prompt budget.
- Recent LLM robotic planners if implementation effort is reasonable.

Most practical option:

Use internal ablations as stronger baselines:

- Full ConceptBot.
- No-KG.
- No-OPE.
- No-URP.
- No-dynamic-properties for the new case study.

This may not fully satisfy the "recent baselines" request, but it strengthens the experimental section with minimal implementation risk.

### 6. Planner Voting `n`

Reviewer concern:

The planner uses `n = 5` samples, but there is no analysis of how performance changes with different `n`.

Current response:

Not addressed.

Recommended experiment:

Run a small ablation with:

```text
n = 1, 3, 5, 7, 9
```

Report:

- success rate;
- invalid plan rate;
- average number of LLM calls;
- latency or cost.

Suggested conclusion:

Show whether `n = 5` is a good tradeoff between robustness and overhead. If performance saturates at `n = 5`, that justifies the choice.

### 7. Real-World Failure Cases

Reviewer concern:

The paper claims improved safety but does not discuss enough real-world failures.

Current response:

Not addressed by code.

Recommended addition:

Add a subsection:

```text
Qualitative Failure Cases and Limitations
```

Useful failure categories:

- ConceptNet missing or noisy relations.
- LLM infers properties without enough evidence.
- Dynamic property induction proposes useful properties, but OPE labels them incorrectly.
- Correct semantic plan fails due to grasping or placement physics.
- Ambiguous object names cause incorrect relation retrieval.

This is especially important because the microwave demo already revealed a valuable failure: aluminum tray can be mislabeled as microwave safe without strong material evidence.

### 8. Computational Overhead

Reviewer concern:

OPE and URP add non-negligible overhead, and real-time feasibility is not discussed.

Current response:

Not addressed.

Recommended measurement:

Add a small timing script or log table measuring:

- ConceptNet retrieval time;
- embedding/similarity filtering time;
- OPE LLM time;
- URP LLM time;
- Planner LLM time;
- total cold-cache time;
- total warm-cache time.

Report mean and standard deviation over a small set of instructions.

Suggested discussion:

- OPE/URP are not hard real-time modules.
- Cache hits reduce repeated ConceptNet/embedding overhead.
- Smaller LLMs and local caches can reduce latency.
- Dynamic Property Induction is disabled by default because it adds another LLM call.

### 9. No-KG Ablation With CoT

Reviewer concern:

The No-KG ablation keeps the Chain-of-Thought prompt template, allowing the LLM to compensate with internal commonsense. This may underestimate the KG contribution.

Current response:

Not addressed.

Recommended experiment:

Add a fairer ablation table:

- Full ConceptBot.
- No-KG with same CoT prompt.
- No-KG without CoT or without KG-specific reasoning scaffold.
- KG-only evidence without OPE/URP if feasible.

Suggested interpretation:

The existing No-KG result tests whether ConceptNet adds value over a strong LLM reasoning baseline. A stricter No-KG-no-CoT variant tests how much the prompt scaffold itself contributes.

## Recommended Revision Package

### Must Have

1. Add Dynamic Property Induction description and qualitative case study.
2. Add theta sensitivity analysis plot/table.
3. Add explicit limitations for semantic risk vs physical risk.
4. Add real-world failure cases.
5. Add runtime overhead table.
6. Add response letter mapping every reviewer point to paper changes.
7. Upload experimental video in mp4 format.

### Strongly Recommended

1. Add voting `n` ablation.
2. Add improved No-KG ablation.
3. Add at least one additional baseline or stronger internal ablation.
4. Add discussion of implicit risk detection and either implement a trigger or state it as future work.

### Optional Code Extensions

These are useful but not required before resubmission if time is limited:

- adaptive theta;
- implicit risk trigger;
- dynamic properties in material/risk OPE variants;
- cached dynamic-property induction for sweeps;
- physical risk score integration.

## Suggested Paper Locations

| Paper Section | Suggested Change |
| --- | --- |
| Method / OPE | Add Dynamic Property Induction as optional extension |
| Method / OPE | Clarify category-specific target properties and prompts |
| Experiments | Add theta sweep sensitivity analysis |
| Experiments | Add voting `n` ablation |
| Experiments | Add No-KG prompt-control ablation |
| Experiments | Add runtime overhead table |
| Real-world Results | Add qualitative failure cases |
| Discussion | Add semantic-vs-physical risk limitation |
| Supplementary Material | Add full prompts, category pipeline table, videos |

## Suggested Cover Letter Framing

For the OPE attribute-space comment:

```text
We thank the reviewer for pointing out that the original OPE module relied on a manually predefined attribute space. We addressed this by adding an optional Dynamic Property Induction step that derives task-relevant properties from the user instruction and detected objects, then merges them with the original OPE target properties before ConceptNet filtering. We added a microwave-heating case study showing that the system can induce properties such as microwave-safe, heat-resistant, non-metallic, and container type.
```

For the theta comment:

```text
We added a systematic sensitivity analysis over the ConceptNet semantic filtering threshold theta. The revised manuscript reports performance trends across task categories and discusses the recall-noise tradeoff that motivated the selected operating point.
```

For the risk limitation:

```text
We clarified that the current Risk Index captures semantic commonsense risk, not full physical execution risk. We added a discussion of physical hazards such as unstable placements, thermal deformation, and grasp failure, and identify physics-aware risk estimation as future work.
```

For the overhead comment:

```text
We added runtime measurements separating ConceptNet retrieval, OPE, URP, and planner calls, and discuss cache-based optimization and real-time feasibility.
```

## Priority Order

If time is limited before the May 19, 2026 revision deadline, prioritize:

1. Theta sweep results, because both reviewers ask for this.
2. Dynamic Property Induction case study, because it directly addresses a major methodological comment.
3. Runtime overhead table, because it is usually easy to measure and improves transparency.
4. Failure cases and semantic-risk limitation, because they reduce overclaiming.
5. No-KG and voting ablations, because they strengthen the experimental section.
6. Additional baselines, if implementation time allows.
