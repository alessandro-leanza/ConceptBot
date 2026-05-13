# Physical Risk Limitations Revision

This document addresses the Reviewer 1 comment that the Risk Index is derived mainly from semantic knowledge rather than grounded physical reasoning.

Reviewer comment:

```text
The Risk Index is primarily derived from semantic knowledge, such as fragility, toxicity, and hazardous interactions, rather than grounded physical reasoning (e.g., physics simulation, contact modeling, thermal effects, or stability analysis). Consequently, while the system can avoid many common-sense hazards, it may still lack the ability to capture finer-grained physical risks in real robotic execution, such as unstable placement configurations or failure due to center-of-mass shifts.
```

## Core Position

The correct response is to acknowledge the limitation directly.

ConceptBot's Risk Index should be framed as:

- a semantic and commonsense risk layer;
- useful for high-level LLM planning failures caused by missing object/property/interaction knowledge;
- complementary to low-level physical validation;
- not a physics-based safety guarantee.

Do not claim that the current Risk Index models:

- contact forces;
- center of mass;
- support polygons;
- geometry-dependent stability;
- thermal transfer;
- deformability under load or temperature;
- force/torque execution safety.

## Recommended Manuscript Changes

### 1. Add Clarification In Section 3.1

Recommended location:

- Section 3.1, immediately after the Risk Index definition and 1--5 scoring explanation.

LaTeX text:

```latex
\paragraph{Scope of the Risk Index.}
The Risk Index should be interpreted as a semantic risk estimate derived from commonsense knowledge, rather than as a physics-based safety certificate. It captures risks that can be inferred from object categories, properties, and pairwise commonsense interactions, such as fragility, toxicity, sharpness, heat-related hazards, or unsafe object combinations. However, it does not model contact forces, geometry, support polygons, center of mass, thermal transfer, deformation, or dynamic stability. Therefore, the Risk Index is intended to complement, rather than replace, low-level geometric and physical validation mechanisms.
```

Shorter version if space is limited:

```latex
The Risk Index provides semantic-level safety awareness rather than a physics-based safety guarantee. It captures commonsense hazards and risky object interactions, but does not model contact forces, geometry, center of mass, thermal transfer, or dynamic stability. It should therefore be viewed as complementary to low-level physical validation.
```

### 2. Add Failure-Case Discussion

Recommended location:

- End of Results section, after risk-aware experiments; or
- Discussion / Limitations section if space is limited.

LaTeX text:

```latex
\paragraph{Representative limitations of semantic risk reasoning.}
Although the Risk Index improves commonsense safety awareness, some failure modes require physical grounding beyond semantic knowledge. For example, the system may recognize that glass objects are fragile or that stacking objects can be risky, but it does not compute the support polygon, contact surface, center of mass, or stability margin of a specific placement. Similarly, it may identify that hot liquid or microwave heating is semantically safety-relevant, but it does not simulate thermal transfer, container deformation, or spill dynamics. These cases illustrate that semantic risk reasoning is useful for high-level planning, but must be combined with geometric and physics-based checks for execution-time safety.
```

### 3. Add Failure-Case Table

Recommended location:

- Results discussion or supplementary material.

LaTeX table:

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

If the paper is tight on space, use only the paragraph and move the table to supplementary material.

### 4. Add Limitations Paragraph

Recommended location:

- Section 6, Conclusion and Limitations.

LaTeX text:

```latex
\paragraph{Physical grounding of risk.}
A further limitation concerns the physical grounding of risk. The proposed Risk Index improves semantic safety awareness by identifying commonsense hazards and risky object interactions, but it does not replace low-level physical validation. Fine-grained execution risks, such as unstable placements, center-of-mass shifts, contact dynamics, thermal effects, grasp-induced failures, or collisions during placement, require geometric and physics-based reasoning beyond the current scope. Integrating the semantic Risk Index with simulation, stability analysis, force/torque feedback, and closed-loop execution monitoring is an important direction for future work.
```

## Response Letter Text

Use this in the response to Reviewer 1:

```latex
\textbf{Comment:} The Risk Index is primarily derived from semantic knowledge, such as fragility, toxicity, and hazardous interactions, rather than grounded physical reasoning (e.g., physics simulation, contact modeling, thermal effects, or stability analysis). Consequently, while the system can avoid many common-sense hazards, it may still lack the ability to capture finer-grained physical risks in real robotic execution, such as unstable placement configurations or failure due to center-of-mass shifts.

\textcolor{NavyBlue}{We thank the reviewer for this insightful and important observation. We fully agree that the current Risk Index operates at a semantic level, relying on commonsense object properties and object-interaction knowledge, rather than on explicit physical reasoning such as physics simulation, contact modeling, thermal transfer, or stability analysis.}

\textcolor{NavyBlue}{Our goal in this work is to address a complementary aspect of safety in LLM-based planning: commonsense-aware risk mitigation. Many LLM-based planners fail because they lack high-level knowledge about object properties and unsafe object combinations, such as selecting an inappropriate container for hot liquid, placing fragile objects in risky configurations, or combining objects with incompatible affordances. The Risk Index is designed to reduce such semantic planning failures.}

\textcolor{NavyBlue}{However, as correctly pointed out by the reviewer, this semantic approach does not fully capture fine-grained physical risks, such as unstable placements, contact dynamics, thermal effects, or center-of-mass shifts. We revised the manuscript to make this distinction explicit. In particular, we clarify that the Risk Index provides semantic-level safety awareness rather than a complete physical safety guarantee, and that it should be viewed as a complementary layer to low-level geometric and physical validation mechanisms.}

\textcolor{NavyBlue}{We also added a discussion of representative cases where semantic reasoning alone is insufficient, including object stacking, hot-liquid handling, microwave/material compatibility, and breakage-sensitive placement. These examples highlight the need for tighter integration between commonsense semantic knowledge and physical reasoning. Finally, we added this point to the limitations and future-work discussion, noting that future versions should integrate semantic risk estimation with stability checks, simulation, force/torque feedback, and closed-loop execution monitoring.}
```

## What This Addresses

This response addresses the reviewer by:

- explicitly accepting the limitation;
- narrowing the claim of the Risk Index to semantic safety awareness;
- explaining why semantic risk reasoning is still valuable;
- adding representative failure cases;
- identifying concrete future integration paths.

## What It Does Not Address

This response does not solve physical safety reasoning in the codebase.

It does not add:

- physics simulation;
- real contact modeling;
- stability analysis;
- center-of-mass estimation;
- thermal modeling;
- force/torque validation.

That is acceptable as long as the paper does not overclaim. The revised manuscript should clearly state that ConceptBot's current contribution is semantic risk mitigation, not full physical execution safety.

## Recommended Final Position

Use this sentence as the high-level framing:

```latex
ConceptBot improves commonsense-aware planning safety, but it should be integrated with physical validation modules before being treated as a complete execution-time safety system.
```
