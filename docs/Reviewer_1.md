# Reviewer 1 Response Package

This document consolidates the response to the four methodological comments from Reviewer 1. It includes the rebuttal text, the exact manuscript additions/modifications, and the supporting tables/figures to insert.

## Comment 1: OPE Uses A Predefined Attribute Space

### Reviewer Comment

```text
The OPE module relies on a predefined set of target attributes for filtering, such as fragile, hold liquid, dangerous. In material-specific tasks, this attribute set is replaced by categories such as glass, plastic, and metal. This design implies that the attribute space is manually specified a priori. As a result, when a task requires additional relevant properties (e.g., heat resistance or microwave compatibility), the proposed method may fail to explicitly extract them, leading to incomplete object understanding.
```

### Response Letter Text

```latex
\textcolor{NavyBlue}{We thank the reviewer for highlighting that the original OPE module relied on a manually predefined target-attribute space. In the revised system, we added Dynamic Property Induction, a task-conditioned property-space expansion mechanism. Given the user instruction, detected objects, and the base OPE property list, the system proposes a small capped set of task-relevant properties, normalizes and deduplicates them, and merges them with the original OPE properties before ConceptNet semantic filtering. Thus, the base schema is preserved for reproducibility, but OPE can explicitly consider additional task-specific properties such as microwave-safe, heat-resistant, non-metallic, waterproof, food-safe, and chemically reactive.}

\textcolor{NavyBlue}{We also added a lightweight need assessor that predicts when the base OPE schema is insufficient. In a curated audit of missing-property and base-schema-sufficient cases, the assessor achieved 100\% trigger accuracy, and the induced properties covered 100\% of the expected missing task properties in the positive cases. We added the microwave-heating case study, the audit table, and limitations noting that induced properties can still be noisy and require evidence validation.}
```

### Manuscript Changes

Add this in **Section 3.1, Object Properties Extraction**, immediately after the paragraph that introduces semantic filtering or after the paragraph listing the base target properties:

```latex
\paragraph{Dynamic Property Induction.}
The base OPE schema uses a compact set of predefined properties to keep the output stable and comparable across tasks. However, some instructions require task-specific attributes that are not part of this base list, such as microwave compatibility, heat resistance, waterproofness, or food safety. To address this limitation, we add an optional Dynamic Property Induction step before ConceptNet semantic filtering. Given the user instruction, the detected objects, and the base OPE properties, an LLM proposes a small capped set of additional task-relevant properties. These properties are normalized, deduplicated against the base properties, and appended to the OPE target list. ConceptNet relations are then retrieved and filtered against the merged property set, and the OPE prompt asks the LLM to infer both the original base properties and the induced task-specific properties.

The base properties are never removed, so the original OPE behavior remains reproducible when dynamic induction is disabled. If the induction call fails or returns invalid JSON, the system safely falls back to the predefined property set. In the current implementation, dynamic induction is integrated into generic OPE only; extending material-specific and risk-specific OPE variants is left for future work.

\paragraph{Dynamic-property need assessment.}
To avoid enabling dynamic induction for every instruction, we add a lightweight assessor that predicts whether the base OPE schema is insufficient for the current task. The assessor examines the user instruction, detected objects, and base properties, and returns a JSON decision indicating whether Dynamic Property Induction is required. It is designed to activate only when additional task-specific properties could change object selection, destination selection, or task success, rather than when extra labels would merely be interesting.
```

Add this qualitative example in **Section 5 / Results** or in a short supplementary case-study subsection:

```latex
\paragraph{Microwave-heating case study.}
As a qualitative example, consider the instruction ``Heat my food in the microwave'' with detected objects \{aluminum tray, glass container, soup bowl, microwave oven\}. The original OPE property list contains general attributes such as \textit{dangerous}, \textit{fragile}, \textit{hold liquid}, \textit{safe}, \textit{stable}, and \textit{poisonous}, but it does not explicitly ask whether an object is microwave-compatible. Dynamic Property Induction adds task-relevant properties such as \textit{microwave-safe}, \textit{heat-resistant}, \textit{non-metallic}, and \textit{container type}. These induced properties are then included in the OPE relation-filtering targets and in the OPE output schema, allowing the system to explicitly reason about properties that were absent from the manually predefined list.
```

Add this table in **Section 5 / Results** or Supplementary Material:

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
\hline
\end{tabular}
\end{table}
```

Add this audit table in **Section 5 / Results** or Supplementary Material:

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

Add this sentence near the audit table:

```latex
The audit evaluates whether the system can detect when the base OPE property schema is insufficient and whether the induced properties cover the expected missing task properties. It does not measure full planner success; rather, it isolates the property-space expansion mechanism targeted by the reviewer comment.
```

Add this in **Appendix C.1, OPE prompt templates**:

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

Add this limitation in **Section 6**:

```latex
Dynamic Property Induction also introduces new limitations. Because induced properties are proposed by an LLM, they can be noisy, overly broad, or inconsistently named. We therefore cap the number of induced properties, normalize and deduplicate them against the base OPE schema, and disable the mechanism by default to preserve reproducibility. Moreover, an induced property only adds an explicit reasoning target; it does not guarantee that ConceptNet contains sufficient evidence to infer the property correctly. Future work will integrate stricter evidence validation and extend dynamic properties to material- and risk-specific OPE variants.
```

### Supporting Files

- `scripts/modules/dynamic_properties.py`
- `scripts/experiments/dynamic_properties/eval_dynamic_properties.py`
- `scripts/experiments/dynamic_properties/results/dynamic_property_audit_v5.json`
- `docs/DYNAMIC_PROPERTY_REVISION_INSERTIONS.md`

## Comment 2: Fixed Similarity Threshold `theta = 0.75`

### Reviewer Comment

```text
The cosine similarity threshold used for semantic filtering is fixed at 0.75, based on empirical observations to balance recall and precision. While this heuristic may work for certain objects, it does not generalize across all object categories and task contexts. In particular, for long-tail entities, a higher threshold may discard useful relations, whereas for common entities, a lower threshold may introduce noisy or weakly relevant relations.
```

### Response Letter Text

```latex
\textcolor{NavyBlue}{We thank the reviewer for pointing out that a fixed semantic-filtering threshold may not generalize across categories and long-tail entities. In the revised manuscript, we no longer present $\theta=0.75$ as an empirically chosen task-performance optimum. Instead, we added a retrieval-sensitivity analysis over $\theta \in \{0.65, 0.70, 0.75, 0.80, 0.85\}$. The analysis reports retained-relation ratio, OPE zero-object ratio, and URP zero-keyword ratio. These diagnostics directly measure the KG coverage/noise tradeoff induced by $\theta$.}

\textcolor{NavyBlue}{The results show that low thresholds retain nearly all relations, while stricter thresholds sharply increase zero-relation cases; $\theta=0.75$ preserves substantial KG coverage while filtering low-similarity evidence. We therefore describe $\theta=0.75$ as a practical retrieval operating point rather than a universally optimal threshold. We also added a limitation noting that future work should replace the fixed global threshold with adaptive retrieval or top-$k$ fallback for long-tail concepts.}
```

### Manuscript Changes

Replace the threshold justification in **Section 3.1, Semantic Filtering** with:

```latex
Relevance is measured via cosine similarity, and relations exceeding a threshold $\theta$ are retained:
$R_{\mathrm{fil}} = \{r \in R_i : \mathrm{similarity}(v_r, v_{\mathrm{prop}}) > \theta\}$. Since no global threshold can be universally optimal for all ConceptNet entities, we analyze $\theta$ as a retrieval parameter rather than as a direct task-performance hyperparameter. Specifically, $\theta$ controls the tradeoff between relation coverage and semantic noise. Lower values retain broad commonsense coverage but include weakly related triples, whereas higher values suppress noise but can remove all KG evidence for long-tail objects or task-specific concepts.

We select $\theta=0.75$ as the default operating point because it preserves substantial ConceptNet coverage while filtering a meaningful fraction of low-similarity relations. In our threshold sweep over $\theta \in \{0.65, 0.70, 0.75, 0.80, 0.85\}$, $\theta=0.75$ retains approximately 81\% of candidate relations on average, while stricter thresholds produce a severe coverage collapse: at $\theta=0.80$ the retained-relation ratio drops to approximately 14\%, and at $\theta=0.85$ to approximately 4\%. This collapse also increases zero-relation failures in both OPE objects and URP keywords, forcing the downstream LLM to rely more heavily on its internal prior knowledge rather than KG-grounded evidence.
```

Add this in **Section 4.3, Performance Metrics / Experimental Protocol**:

```latex
\paragraph{Threshold retrieval analysis.}
To examine the robustness of ConceptNet semantic filtering, we perform a retrieval-sensitivity sweep over $\theta \in \{0.65, 0.70, 0.75, 0.80, 0.85\}$. For each threshold, we record retrieval diagnostics rather than using task success as the primary criterion: the fraction of ConceptNet relations retained after filtering, the fraction of OPE objects with no retained relations, and the fraction of URP keywords with no retained relations. These diagnostics directly measure the threshold's effect on KG coverage and are more appropriate for selecting $\theta$ than end-to-end success alone, since policy success can also be affected by URP, the planner, and the LLM's ability to compensate without KG evidence.
```

Add this in **Section 5, Results**, as a short subsection:

```latex
\paragraph{Sensitivity of ConceptNet Retrieval to $\theta$.}
We further analyze how the semantic-filtering threshold affects KG evidence available to OPE and URP. Figure~X shows that $\theta$ strongly controls relation coverage. At $\theta=0.65$ and $\theta=0.70$, nearly all candidate relations are retained, which maximizes recall but risks injecting weakly related triples into the LLM context. At $\theta=0.75$, the system still retains approximately 81\% of candidate relations while filtering a meaningful portion of low-similarity evidence. In contrast, $\theta \geq 0.80$ causes a sharp coverage collapse: the retained-relation ratio falls to approximately 14\% at $\theta=0.80$ and 4\% at $\theta=0.85$.

The zero-relation diagnostics are particularly important for long-tail concepts. At $\theta=0.80$ and $\theta=0.85$, many object and keyword queries retain no ConceptNet evidence. In such cases, downstream reasoning becomes less KG-grounded and more dependent on the LLM's internal commonsense. Therefore, we retain $\theta=0.75$ as the default because it avoids the severe zero-relation regime while still reducing noise compared with lower thresholds. This analysis does not imply that $\theta=0.75$ maximizes success for every task category; rather, it supports $\theta=0.75$ as a stable retrieval setting for maintaining KG-grounded evidence across categories.
```

Add this table in **Section 5**:

```latex
\begin{table}[t]
\centering
\caption{Retrieval sensitivity of ConceptNet semantic filtering to $\theta$. Kept ratio is the fraction of candidate ConceptNet relations retained after semantic filtering.}
\label{tab:theta-retrieval}
\begin{tabular}{lccc}
\hline
$\theta$ & Kept ratio & OPE zero-object ratio & URP zero-keyword ratio \\
\hline
0.65 & 1.000 & 0.276 & 0.077 \\
0.70 & 0.999 & 0.276 & 0.077 \\
0.75 & 0.812 & 0.352 & 0.077 \\
0.80 & 0.142 & 0.812 & 0.432 \\
0.85 & 0.040 & 0.955 & 0.958 \\
\hline
\end{tabular}
\end{table}
```

Add figure **`scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png`** in Section 5 with caption:

```latex
\caption{Effect of $\theta$ on ConceptNet retrieval coverage. The plot reports the retained-relation ratio, the OPE zero-object ratio, and the URP zero-keyword ratio. Stricter thresholds sharply increase the fraction of objects and request keywords with no retained KG evidence, especially at $\theta \geq 0.80$. The default $\theta=0.75$ avoids this coverage collapse while still filtering low-similarity relations.}
```

Add this limitation in **Section 6**:

```latex
The semantic-filtering threshold is still global. Although the retrieval sweep supports $\theta=0.75$ as a practical operating point, a single threshold cannot be optimal for all object names, relation types, and task contexts. Future work will investigate adaptive retrieval strategies, such as object-specific thresholds, relation-type-aware thresholds, or top-$k$ fallback when strict filtering removes all ConceptNet evidence.
```

### Supporting Files

- `docs/THETA_REVISION_INSERTIONS.md`
- `scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png`
- `scripts/experiments/theta/results/threshold_sweep_combined.csv`

## Comment 3: Risk Index Activated Only When User Explicitly Requests Safety

### Reviewer Comment

```text
The Risk Index-enhanced version of OPE is invoked only when users explicitly require enhanced safety. This raises a potential concern: if the user does not explicitly specify safety requirements, but the task inherently involves risk, the system may not activate the full risk evaluation mechanism. This could lead to insufficient awareness of implicit safety hazards.
```

### Response Letter Text

```latex
\textcolor{NavyBlue}{We thank the reviewer for noting that risk-aware OPE should not depend solely on an explicit user request for safety. In the revised implementation, we add a lightweight implicit-risk trigger that evaluates whether the numerical, interaction-aware Risk Index should be activated from the user instruction and detected objects. The trigger is evaluated in an LLM-only setting where category overrides are disabled and the category label is hidden from the trigger.}

\textcolor{NavyBlue}{On the current ConceptBot instruction set, it correctly activates the Risk Index for all risk-aware tasks and does not activate it for the explicit or implicit non-risk tasks (35/35 correct decisions; 100\% risk-aware recall; 0\% explicit false-positive rate). We also clarify in the manuscript that this trigger is not a complete physical safety monitor and does not replace grounded execution-time safety checks.}
```

### Manuscript Changes

Replace the problematic sentence in **Section 3.1, Risk Index**:

```text
This version of the OPE module focuses only on evaluating object risk levels rather than extracting all properties, and is invoked only when users require enhanced safety.
```

with:

```latex
This version of OPE focuses on evaluating object risk levels and hazardous object interactions rather than extracting the full generic property schema. In the current implementation, it is used for safety-sensitive task categories and explicitly risk-aware scenarios. More generally, activating this mode requires a task-mode selection policy: the system should identify when an instruction or scene contains implicit hazards even if the user does not explicitly ask for safety.
```

Add this paragraph after the Risk Index description in **Section 3.1**:

```latex
\paragraph{Implicit Risk Trigger.}
To address cases where safety-relevant reasoning may be required even without an explicit user request for safety, we added a lightweight implicit-risk trigger before selecting the OPE/URP mode. The trigger evaluates the user instruction and detected objects and predicts whether the numerical, interaction-aware Risk Index should be activated. Importantly, the trigger is not a generic hazard detector: it activates the Risk Index only when enhanced risk reasoning could change the robot's choice of object, destination, or object--object interaction. The trigger uses an LLM-based JSON decision followed by conservative validation of core risk cues, such as heat, microwave/material compatibility, sharp tools, vulnerable recipients, fragile stacking, breakage-sensitive appliance use, chemicals, toxicity, electrical risk, or unstable placement.
```

Add this table in **Section 5 / Results** or Supplementary Material:

```latex
\begin{table}[t]
\centering
\caption{Implicit Risk Index trigger audit. The LLM-only setting hides the instruction category from the trigger and disables category overrides.}
\label{tab:risk-trigger-audit}
\begin{tabular}{lcccc}
\hline
\textbf{Category} & \textbf{Items} & \textbf{Expected Risk} & \textbf{Triggered} & \textbf{Accuracy} \\
\hline
Explicit unambiguous & 7  & 0 & 0 & 100\% \\
Explicit ambiguous   & 10 & 0 & 0 & 100\% \\
Implicit             & 10 & 0 & 0 & 100\% \\
Risk-aware           & 8  & 8 & 8 & 100\% \\
\hline
\textbf{Overall}     & 35 & 8 & 8 & 100\% \\
\hline
\end{tabular}
\end{table}
```

Add this caution near the table:

```latex
The audit labels the current explicit and implicit splits as non-risk-trigger cases because these instructions require semantic disambiguation or ordinary object selection, while the risk-aware split contains tasks where risk reasoning changes the choice of object, destination, or interaction. This audit does not claim complete detection of all future implicit hazards; it verifies that the trigger behaves correctly on the current ConceptBot instruction set.
```

Add this limitation in **Section 6**:

```latex
The Risk Index still depends on task-mode selection. In this revision, we add a lightweight implicit-risk trigger and validate it on the current ConceptBot instruction set. However, the trigger is not a complete safety monitor: it only estimates whether semantic, interaction-aware risk reasoning should be activated before planning. It does not replace physical risk validation, force/torque monitoring, geometric stability analysis, thermal modeling, or closed-loop execution monitoring.
```

### Supporting Files

- `scripts/modules/risk_trigger.py`
- `scripts/experiments/risk_trigger/eval_risk_trigger.py`
- `scripts/experiments/risk_trigger/results/risk_trigger_llm_only_v7.json`
- `docs/RISK_INDEX_ACTIVATION_REVISION_PLAN.md`

## Comment 4: Risk Index Is Semantic, Not Grounded Physical Reasoning

### Reviewer Comment

```text
The Risk Index is primarily derived from semantic knowledge, such as fragility, toxicity, and hazardous interactions, rather than grounded physical reasoning (e.g., physics simulation, contact modeling, thermal effects, or stability analysis). Consequently, while the system can avoid many common-sense hazards, it may still lack the ability to capture finer-grained physical risks in real robotic execution, such as unstable placement configurations or failure due to center-of-mass shifts.
```

### Response Letter Text

```latex
\textcolor{NavyBlue}{We thank the reviewer for this insightful and important observation. We fully agree that the current Risk Index operates at a semantic level, relying on commonsense object properties and object-interaction knowledge, rather than on explicit physical reasoning such as physics simulation, contact modeling, thermal transfer, or stability analysis.}

\textcolor{NavyBlue}{Our goal in this work is to address a complementary aspect of safety in LLM-based planning: commonsense-aware risk mitigation. Many LLM-based planners fail because they lack high-level knowledge about object properties and unsafe object combinations, such as selecting an inappropriate container for hot liquid, placing fragile objects in risky configurations, or combining objects with incompatible affordances. The Risk Index is designed to reduce such semantic planning failures.}

\textcolor{NavyBlue}{However, as correctly pointed out by the reviewer, this semantic approach does not fully capture fine-grained physical risks, such as unstable placements, contact dynamics, thermal effects, or center-of-mass shifts. We revised the manuscript to make this distinction explicit. In particular, we clarify that the Risk Index provides semantic-level safety awareness rather than a complete physical safety guarantee, and that it should be viewed as a complementary layer to low-level geometric and physical validation mechanisms.}

\textcolor{NavyBlue}{We also added a discussion of representative cases where semantic reasoning alone is insufficient, including object stacking, hot-liquid handling, microwave/material compatibility, and breakage-sensitive placement. These examples highlight the need for tighter integration between commonsense semantic knowledge and physical reasoning. Finally, we added this point to the limitations and future-work discussion, noting that future versions should integrate semantic risk estimation with stability checks, simulation, force/torque feedback, and closed-loop execution monitoring.}
```

### Manuscript Changes

Add this after the Risk Index definition in **Section 3.1**:

```latex
\paragraph{Scope of the Risk Index.}
The Risk Index should be interpreted as a semantic risk estimate derived from commonsense knowledge, rather than as a physics-based safety certificate. It captures risks that can be inferred from object categories, properties, and pairwise commonsense interactions, such as fragility, toxicity, sharpness, heat-related hazards, or unsafe object combinations. However, it does not model contact forces, geometry, support polygons, center of mass, thermal transfer, deformation, or dynamic stability. Therefore, the Risk Index is intended to complement, rather than replace, low-level geometric and physical validation mechanisms.
```

Add this in **Results Discussion** or **Section 6**:

```latex
\paragraph{Representative limitations of semantic risk reasoning.}
Although the Risk Index improves commonsense safety awareness, some failure modes require physical grounding beyond semantic knowledge. For example, the system may recognize that glass objects are fragile or that stacking objects can be risky, but it does not compute the support polygon, contact surface, center of mass, or stability margin of a specific placement. Similarly, it may identify that hot liquid or microwave heating is semantically safety-relevant, but it does not simulate thermal transfer, container deformation, or spill dynamics. These cases illustrate that semantic risk reasoning is useful for high-level planning, but must be combined with geometric and physics-based checks for execution-time safety.
```

Add this table in **Results Discussion** or Supplementary Material:

```latex
\begin{table}[t]
\centering
\caption{Representative limitations of semantic-only Risk Index reasoning.}
\label{tab:semantic-risk-limitations}
\begin{tabular}{p{0.27\linewidth}p{0.31\linewidth}p{0.32\linewidth}}
\hline
\textbf{Scenario} & \textbf{Captured by Risk Index} & \textbf{Missing physical reasoning} \\
\hline
Sharp object near a child &
Commonsense hazard of sharp tools and unsafe recipient &
Exact grasp pose, handover geometry, and execution-time proximity \\
\hline
Hot liquid handling &
Semantic hazard of hot liquid and unsuitable containers &
Temperature, spill dynamics, deformation, and contact safety \\
\hline
Object stacking &
Fragility and instability cues from object properties &
Center of mass, support polygon, contact surface, and stability margin \\
\hline
Dishwasher or handwashing assignment &
Breakage and material-compatibility cues &
Appliance geometry, contact forces, and collision during placement \\
\hline
Microwave heating &
Material compatibility and unsafe object combinations &
Thermal transfer, arcing behavior, and container deformation \\
\hline
\end{tabular}
\end{table}
```

Add this limitation in **Section 6**:

```latex
\paragraph{Physical grounding of risk.}
A further limitation concerns the physical grounding of risk. The proposed Risk Index improves semantic safety awareness by identifying commonsense hazards and risky object interactions, but it does not replace low-level physical validation. Fine-grained execution risks, such as unstable placements, center-of-mass shifts, contact dynamics, thermal effects, grasp-induced failures, or collisions during placement, require geometric and physics-based reasoning beyond the current scope. Integrating the semantic Risk Index with simulation, stability analysis, force/torque feedback, and closed-loop execution monitoring is an important direction for future work.
```

### Supporting Files

- `docs/PHYSICAL_RISK_LIMITATIONS_REVISION.md`

## Final Checklist For Reviewer 1

- Add Dynamic Property Induction method text in Section 3.1.
- Add Dynamic Property microwave case study and curated audit table.
- Replace theta threshold justification and add retrieval-sensitivity figure/table.
- Add implicit Risk Trigger method text and audit table.
- Add semantic-vs-physical Risk Index scope clarification.
- Add physical-risk limitation paragraph and optional failure-case table.
- In the response letter, include exact manuscript locations after final figure/table numbering.

## Recommended Claim Boundaries

Use these boundaries consistently:

- Dynamic Property Induction expands the property space, but does not guarantee perfect factual inference.
- `theta=0.75` is a retrieval operating point, not a universal task-success optimum.
- The implicit risk trigger is validated on the current instruction set, not a complete hazard detector.
- The Risk Index is semantic-level safety awareness, not physical execution-time safety validation.
