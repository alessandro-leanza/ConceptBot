# Dynamic Property Induction Revision Insertions

This document explains where and how to revise `docs/ConceptBot_v1.pdf` to address the reviewer concern that OPE relies on a manually predefined attribute space.

Reviewer comment addressed:

```text
The OPE module relies on a predefined set of target attributes for filtering, such as fragile, hold liquid, dangerous. In material-specific tasks, this attribute set is replaced by categories such as glass, plastic, and metal. This design implies that the attribute space is manually specified a priori. As a result, when a task requires additional relevant properties (e.g., heat resistance or microwave compatibility), the proposed method may fail to explicitly extract them, leading to incomplete object understanding.
```

Implemented repository support:

- `scripts/modules/dynamic_properties.py`
- `scripts/experiments/demo_dynamic_properties.py`
- `scripts/experiments/dynamic_properties/eval_dynamic_properties.py`
- `scripts/modules/ope.py`
- `docs/DYNAMIC_PROPERTY_INDUCTION_DESIGN.md`
- `docs/DYNAMIC_PROPERTY_INDUCTION_USAGE.md`

Important implementation facts:

- Dynamic Property Induction is optional and disabled by default.
- Existing OPE behavior is unchanged when `dynamic_properties=None`.
- The implemented integration is currently limited to generic OPE.
- Material, risk, URP, and theta-sweep integration are intentionally deferred.
- Dynamic properties are appended to base properties; they do not replace predefined OPE properties.
- If LLM induction fails or returns invalid JSON, the function falls back to base properties only.
- The current implementation includes a standalone dynamic-property need assessor and curated audit, but dynamic induction is not enabled by default in the main pipeline.

## Core Position

Use this as the central revised claim:

```text
The original OPE attribute list remains useful as a stable base schema, but it is no longer the only possible attribute space. We add an optional Dynamic Property Induction step that proposes a small set of task-relevant properties from the user instruction and detected objects, then merges them with the predefined OPE properties before ConceptNet relation filtering. This allows OPE to consider task-specific attributes such as microwave-safe, heat-resistant, or non-metallic without changing the default behavior of the original system.
```

Do not overclaim:

- Do not say that dynamic induction solves all missing-property problems.
- Do not say that induced properties are always correct.
- Do not say that this is fully integrated across every OPE variant.

Say:

```text
Dynamic Property Induction is a lightweight extension that reduces dependence on a fixed attribute space while preserving backward compatibility.
```

Add the activation caveat:

```text
The current revision implements both the induction mechanism and a standalone need assessor that evaluates when the base OPE property schema is insufficient. The mechanism remains disabled by default in the main pipeline to preserve reproducibility.
```

## 1. Where To Insert In The Paper

### Section 3.1, OPE

Current location:

- Section 3.1, Object Properties Extraction.
- Insert after the `Semantic Filtering` paragraph and before `Object Properties`, or immediately after the paragraph that lists the base target properties:

```text
The target properties—"fragile," "hold liquid," "dangerous," "safe," "deformable," "stable," and "poisonous"—help assess object characteristics in the robot's context.
```

Why this location:

- The reviewer criticism is specifically about fixed target properties used before relation filtering.
- Dynamic Property Induction changes the target-property set before ConceptNet similarity filtering.
- Inserting it here makes the flow clear.

### Section 4 / Experiments

Add a short qualitative case study after the main task descriptions or in the Results section after the OPE ablation discussion.

Recommended title:

```text
Qualitative Case Study: Dynamic Task-Relevant Properties
```

### Appendix C, OPE Prompt Templates

Add a short prompt addendum showing how dynamic properties are appended only when enabled.

### Section 6, Limitations

Add an honest limitation explaining that induced properties may be noisy and are currently optional.

## 2. Method Text For Section 3.1

### Suggested Plain Text

```text
Dynamic Property Induction. The base OPE schema uses a compact set of predefined properties to keep the output stable and comparable across tasks. However, some instructions require task-specific attributes that are not part of this base list, such as microwave compatibility, heat resistance, waterproofness, or food safety. To address this limitation, we add an optional Dynamic Property Induction step before ConceptNet semantic filtering. Given the user instruction, the detected objects, and the base OPE properties, an LLM proposes a small capped set of additional task-relevant properties. These properties are normalized, deduplicated against the base properties, and appended to the OPE target list. ConceptNet relations are then retrieved and filtered against the merged property set, and the OPE prompt asks the LLM to infer both the original base properties and the induced task-specific properties.

The base properties are never removed, so the original OPE behavior remains reproducible when dynamic induction is disabled. If the induction call fails or returns invalid JSON, the system safely falls back to the predefined property set. In the current implementation, dynamic induction is integrated into generic OPE only; extending material-specific and risk-specific OPE variants is left for future work.
```

Add this short need-assessment paragraph after the method text:

```text
Dynamic-property need assessment. To avoid enabling dynamic induction for every instruction, we add a lightweight assessor that predicts whether the base OPE schema is insufficient for the current task. The assessor examines the user instruction, detected objects, and base properties, and returns a JSON decision indicating whether Dynamic Property Induction is required. It is designed to activate only when additional task-specific properties could change object selection, destination selection, or task success, rather than when extra labels would merely be interesting.
```

### LaTeX Block

```latex
\paragraph{Dynamic Property Induction.}
The base OPE schema uses a compact set of predefined properties to keep the output stable and comparable across tasks. However, some instructions require task-specific attributes that are not part of this base list, such as microwave compatibility, heat resistance, waterproofness, or food safety. To address this limitation, we add an optional Dynamic Property Induction step before ConceptNet semantic filtering. Given the user instruction, the detected objects, and the base OPE properties, an LLM proposes a small capped set of additional task-relevant properties. These properties are normalized, deduplicated against the base properties, and appended to the OPE target list. ConceptNet relations are then retrieved and filtered against the merged property set, and the OPE prompt asks the LLM to infer both the original base properties and the induced task-specific properties.

The base properties are never removed, so the original OPE behavior remains reproducible when dynamic induction is disabled. If the induction call fails or returns invalid JSON, the system safely falls back to the predefined property set. In the current implementation, dynamic induction is integrated into generic OPE only; extending material-specific and risk-specific OPE variants is left for future work.

\paragraph{Dynamic-property need assessment.}
To avoid enabling dynamic induction for every instruction, we add a lightweight assessor that predicts whether the base OPE schema is insufficient for the current task. The assessor examines the user instruction, detected objects, and base properties, and returns a JSON decision indicating whether Dynamic Property Induction is required. It is designed to activate only when additional task-specific properties could change object selection, destination selection, or task success, rather than when extra labels would merely be interesting.
```

## 3. Flow / Figure To Add

A new figure is optional. The mechanism can be explained with a compact text flow.

Recommended figure if space allows:

```text
Instruction + detected objects + base properties
        -> Dynamic Property Induction
        -> dynamic task properties
        -> merge with base OPE properties
        -> ConceptNet retrieval/filtering using merged properties
        -> OPE inference over base + dynamic properties
```

### LaTeX/TikZ-Free Figure Caption

If using a simple diagram or boxed flow:

```latex
\caption{Dynamic Property Induction extends OPE's target-property set before ConceptNet filtering. The induced properties are task-conditioned and appended to the predefined base properties; they do not replace the original OPE schema.}
```

If no figure is added, include the flow as prose in Section 3.1.

## 4. Microwave Case Study

Use this as the main qualitative response to the reviewer example.

Instruction:

```text
Heat my food in the microwave.
```

Detected objects:

```text
aluminum tray, glass container, soup bowl, microwave oven
```

Base OPE properties:

```text
dangerous, fragile, deformable, hold liquid, safe, stable, poisonous
```

Representative induced dynamic properties from the demo:

```text
microwave-safe, heat-resistant, non-metallic, container type
```

Rejected-property example observed in demo runs:

```text
heavy
```

because weight was not central to microwave heating suitability.

Important caveat:

- The exact induced properties may vary slightly across LLM calls.
- Use the result as a qualitative example, not as a deterministic benchmark unless cached/fixed.

### Suggested Plain Text

```text
As a qualitative example, consider the instruction "Heat my food in the microwave" with detected objects {aluminum tray, glass container, soup bowl, microwave oven}. The original OPE property list contains general attributes such as dangerous, fragile, hold liquid, safe, stable, and poisonous, but it does not explicitly ask whether an object is microwave-compatible. Dynamic Property Induction adds task-relevant properties such as microwave-safe, heat-resistant, non-metallic, and container type. These induced properties are then included in the OPE relation-filtering targets and in the OPE output schema, allowing the system to explicitly reason about properties that were absent from the manually predefined list.
```

### LaTeX Block

```latex
\paragraph{Microwave-heating case study.}
As a qualitative example, consider the instruction ``Heat my food in the microwave'' with detected objects \{aluminum tray, glass container, soup bowl, microwave oven\}. The original OPE property list contains general attributes such as \textit{dangerous}, \textit{fragile}, \textit{hold liquid}, \textit{safe}, \textit{stable}, and \textit{poisonous}, but it does not explicitly ask whether an object is microwave-compatible. Dynamic Property Induction adds task-relevant properties such as \textit{microwave-safe}, \textit{heat-resistant}, \textit{non-metallic}, and \textit{container type}. These induced properties are then included in the OPE relation-filtering targets and in the OPE output schema, allowing the system to explicitly reason about properties that were absent from the manually predefined list.
```

## 5. Suggested Case-Study Table

Recommended table title:

```text
Table X. Example of Dynamic Property Induction for a microwave-heating instruction.
```

### Main Paper Or Supplementary Table

| Item | Content |
| --- | --- |
| Instruction | Heat my food in the microwave. |
| Detected objects | aluminum tray; glass container; soup bowl; microwave oven |
| Base OPE properties | dangerous; fragile; deformable; hold liquid; safe; stable; poisonous |
| Induced properties | microwave-safe; heat-resistant; non-metallic; container type |
| Rejected example | heavy |

### LaTeX Table

```latex
\begin{table}[t]
\centering
\caption{Example of Dynamic Property Induction for a microwave-heating instruction.}
\label{tab:dynamic_property_example}
\begin{tabular}{lp{0.68\linewidth}}
\hline
Field & Value \\
\hline
Instruction & Heat my food in the microwave. \\
Detected objects & aluminum tray; glass container; soup bowl; microwave oven \\
Base OPE properties & dangerous; fragile; deformable; hold liquid; safe; stable; poisonous \\
Induced properties & microwave-safe; heat-resistant; non-metallic; container type \\
Rejected example & heavy \\
\hline
\end{tabular}
\end{table}
```

## 6. Curated Audit

To make the response stronger than a single qualitative demo, run a standalone audit:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/dynamic_properties/eval_dynamic_properties.py \
  --model gpt-4o-mini \
  --save-traces \
  --out scripts/experiments/dynamic_properties/results/dynamic_property_audit_v5
```

Final observed metrics:

```text
Trigger accuracy: 100%
Positive trigger recall: 100%
Negative specificity: 100%
Average property recall on positive cases: 100%
Average dynamic-property recall on positive cases: 100%
```

The audit uses five positive missing-property cases and four negative cases where the base OPE schema is sufficient. Positive cases cover microwave compatibility, heat resistance, water protection, food-safe containment, and chemical/food separation. Negative cases cover ordinary object retrieval, ordinary placement, drink retrieval covered by `hold liquid`, and snack retrieval as ordinary semantic selection.

### LaTeX Audit Table

```latex
\begin{table}[t]
\centering
\caption{Dynamic Property Induction audit on curated missing-property cases.}
\label{tab:dynamic-property-audit}
\begin{tabular}{lcccc}
\hline
\textbf{Case type} & \textbf{Items} & \textbf{Expected Dynamic} & \textbf{Triggered} & \textbf{Property Recall} \\
\hline
Missing task-specific properties & 5 & 5 & 5 & 100\% \\
Base schema sufficient & 4 & 0 & 0 & -- \\
\hline
\textbf{Overall trigger accuracy} & 9 & -- & -- & 100\% \\
\hline
\end{tabular}
\end{table}
```

Use this explanatory sentence near the table:

```latex
The audit evaluates whether the system can detect when the base OPE property schema is insufficient and whether the induced properties cover the expected missing task properties. It does not measure full planner success; rather, it isolates the property-space expansion mechanism targeted by the reviewer comment.
```

## 7. Appendix C Prompt Addendum

Current location:

- Appendix C.1, Object Properties Extraction prompt.
- Add after the KG-enabled OPE system message.

### Suggested Text

```text
When Dynamic Property Induction is enabled, the following addendum is appended to the standard OPE system message:
```

### LaTeX Block

```latex
\paragraph{Dynamic-property addendum.}
When Dynamic Property Induction is enabled, the following addendum is appended to the standard OPE system message:

\begin{quote}
Additional task-relevant properties to determine (Yes/No):\\
- PROPERTY\_1 (Yes/No): DESCRIPTION\_1\\
- PROPERTY\_2 (Yes/No): DESCRIPTION\_2\\
\ldots\\
For each object, include one output line for every additional task-relevant property above.
\end{quote}
```

Do not replace the base prompt in Appendix C. The base prompt remains valid for the default system.

## 8. Limitations / Honest Caveats

Current location:

- Section 6, `Conclusion and Limitations`.
- Add after the existing ConceptNet coverage limitation.

### Suggested Plain Text

```text
Dynamic Property Induction also introduces new limitations. Because induced properties are proposed by an LLM, they can be noisy, overly broad, or inconsistently named. We therefore cap the number of induced properties, normalize and deduplicate them against the base OPE schema, and disable the mechanism by default to preserve reproducibility. Moreover, an induced property only adds an explicit reasoning target; it does not guarantee that ConceptNet contains sufficient evidence to infer the property correctly. For example, microwave compatibility may require grounded material and safety knowledge beyond generic commonsense relations. Future work will integrate stricter evidence validation and extend dynamic properties to material- and risk-specific OPE variants.
```

Add this sentence to connect it to the risk-index activation concern:

```text
More broadly, Dynamic Property Induction and risk-aware OPE both require task-mode selection policies. In this revision, we add standalone assessors for both property-space expansion and Risk Index activation, while keeping the default pipeline unchanged for reproducibility.
```

### LaTeX Block

```latex
Dynamic Property Induction also introduces new limitations. Because induced properties are proposed by an LLM, they can be noisy, overly broad, or inconsistently named. We therefore cap the number of induced properties, normalize and deduplicate them against the base OPE schema, and disable the mechanism by default to preserve reproducibility. Moreover, an induced property only adds an explicit reasoning target; it does not guarantee that ConceptNet contains sufficient evidence to infer the property correctly. For example, microwave compatibility may require grounded material and safety knowledge beyond generic commonsense relations. Future work will integrate stricter evidence validation and extend dynamic properties to material- and risk-specific OPE variants.

More broadly, Dynamic Property Induction and risk-aware OPE both require task-mode selection policies. In this revision, we add standalone assessors for both property-space expansion and Risk Index activation, while keeping the default pipeline unchanged for reproducibility.
```

## 9. Response Letter Text

Use this in the reviewer response:

```text
We thank the reviewer for highlighting that the original OPE module relied on a manually predefined target-attribute space. In the revised system, we added Dynamic Property Induction, a task-conditioned property-space expansion mechanism. Given the user instruction, detected objects, and the base OPE property list, the system proposes a small capped set of task-relevant properties, normalizes and deduplicates them, and merges them with the original OPE properties before ConceptNet semantic filtering. Thus, the base schema is preserved for reproducibility, but OPE can explicitly consider additional task-specific properties such as microwave-safe, heat-resistant, non-metallic, waterproof, food-safe, and chemically reactive. We also added a lightweight need assessor that predicts when the base OPE schema is insufficient. In a curated audit of missing-property and base-schema-sufficient cases, the assessor achieved 100% trigger accuracy, and the induced properties covered 100% of the expected missing task properties in the positive cases. We added the microwave-heating case study, the audit table, and limitations noting that induced properties can still be noisy and require evidence validation.
```

Add final manuscript locations after revision:

```text
Section 3.1 Dynamic Property Induction paragraph; Table~\ref{tab:dynamic_property_example}; Table~\ref{tab:dynamic-property-audit}; Appendix C.1 prompt addendum; Section 6 limitations.
```

## 10. What Still Needs To Be Generated Before Final Manuscript

Required:

1. Decide whether the microwave case-study table and audit table go in the main paper or supplement.
2. Assign final figure/table numbers.
3. Update the response letter with exact manuscript locations after editing the manuscript.

Optional but useful:

1. Run the demo once with a fixed model/settings and save the exact JSON output for reproducibility.
2. Add a qualitative before/after OPE prompt excerpt:
   - before: base properties only;
   - after: base properties plus dynamic properties.
3. Add a short note that the current microwave demo may still misclassify some values when ConceptNet evidence is sparse, so the contribution is property-space expansion rather than guaranteed factual correctness.

## 11. Commands For Reproducibility

Run the induction-only demo:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/demo_dynamic_properties.py
```

Run induction and generic OPE:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/demo_dynamic_properties.py --run-ope
```

Use the second command only for qualitative inspection; it calls ConceptNet/cache and the LLM.
