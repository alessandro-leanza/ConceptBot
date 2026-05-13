# Theta Retrieval Analysis Revision Insertions

This document explains how to revise `docs/ConceptBot_v1.pdf` to address the reviewer concern about the fixed ConceptNet semantic-filtering threshold `theta = 0.75`.

The revised argument should **not** claim that `theta = 0.75` is optimal for end-to-end task success. That would be weak, because task success depends on OPE, URP, the planner, LLM behavior, and the task category. Instead, the paper should justify `theta = 0.75` as a **technical retrieval operating point** that balances:

- filtering weak/noisy ConceptNet relations;
- preserving enough relations for complex or long-tail concepts;
- avoiding excessive zero-relation objects and keywords;
- keeping the OPE/URP prompts grounded in KG evidence rather than forcing the LLM to compensate from prior knowledge.

Reviewer comments addressed:

- Reviewer 1: the fixed cosine-similarity threshold may not generalize across object categories and task contexts.
- Reviewer 2.1: the choice of `theta = 0.75` appears empirically chosen; a systematic sensitivity analysis or ablation is needed.

Current theta artifacts:

- `scripts/experiments/theta/results/threshold_sweep_combined.csv`
- `scripts/experiments/theta/results/threshold_sweep_combined_zero_relations.png`
- `scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png`
- per-category JSON/CSV/TXT/PNG files under `scripts/experiments/theta/results/`

Final decisions for the revision:

- Remove theta success-rate analysis from both the main paper and the supplementary material.
- Present the theta sweep as a retrieval diagnostic, not a task-performance ablation.
- Use `theta = 0.75` as a retrieval operating point.
- Keep `theta = 0.75` as the method default.
- Put the main retrieval table in the main paper.
- Put the per-category zero-relation table in the supplementary material.
- Treat adaptive thresholding/top-k fallback as future work, not as a new implemented method.

Important note on trials and caching:

- Repeating the theta sweep 5 or 10 times is useful for task success because the LLM planner can vary across calls.
- It adds little value for the retrieval diagnostics if the extracted objects, keywords, ConceptNet cache, and embedding cache are fixed, because relation filtering at each theta is deterministic over the cached relation/similarity set.
- Repeating the full pipeline without fixed caches could slightly change extracted keywords or LLM-written intermediate outputs, but then the result would mix threshold sensitivity with LLM stochasticity.
- Therefore, for the theta response, one deterministic retrieval sweep is acceptable as long as the paper does not use success rate as the selection criterion.

## Core Position

Use this as the central revised claim:

```text
We do not claim that theta = 0.75 is universally optimal for task success. Instead, we select it as a retrieval operating point for ConceptNet semantic filtering. Lower thresholds retain nearly all ConceptNet relations and therefore admit more weakly related evidence; stricter thresholds remove noise but sharply increase zero-relation cases, especially for complex or long-tail object names. The value theta = 0.75 preserves substantial relation coverage while still filtering a meaningful portion of low-similarity relations.
```

## 1. Replace The Current Threshold Justification In Section 3.1

Current location:

- Section 3.1, Object Properties Extraction, subsection `Semantic Filtering`.
- In the current PDF this is around the paragraph beginning:

```text
To filter useful relationships for our context, we use embeddings...
```

and especially the current sentence:

```text
While there is no universally optimal threshold for filtering ConceptNet relations, we chose theta = 0.75 based on empirical observations to balance relevance and noise.
```

### Problem With Current Text

The current paragraph gives examples and a small coverage observation, but it still sounds heuristic. It does not explicitly say that the threshold was evaluated systematically across task categories. The reviewers are specifically asking for sensitivity analysis, so the revised manuscript should cite a sweep and avoid claiming that `0.75` is universally optimal.

### Suggested Replacement Text

Replace the threshold-justification part of the paragraph with:

```text
Relevance is measured via cosine similarity, and relations exceeding a threshold theta are retained:
R_fil = {r in R_i : similarity(v_r, v_prop) > theta}. Since no global threshold can be universally optimal for all ConceptNet entities, we analyze theta as a retrieval parameter rather than as a direct task-performance hyperparameter. Specifically, theta controls the tradeoff between relation coverage and semantic noise. Lower values retain broad commonsense coverage but include weakly related triples, whereas higher values suppress noise but can remove all KG evidence for long-tail objects or task-specific concepts.

We select theta = 0.75 as the default operating point because it preserves substantial ConceptNet coverage while filtering a meaningful fraction of low-similarity relations. In our threshold sweep over theta in {0.65, 0.70, 0.75, 0.80, 0.85}, theta = 0.75 retains approximately 81% of candidate relations on average, while stricter thresholds produce a severe coverage collapse: at theta = 0.80 the retained-relation ratio drops to approximately 14%, and at theta = 0.85 to approximately 4%. This collapse also increases zero-relation failures in both OPE objects and URP keywords, forcing the downstream LLM to rely more heavily on its internal prior knowledge rather than KG-grounded evidence.
```

Then continue with a shortened example paragraph:

```text
For example, in URP, matching "apple" to "hungry" should retain relations such as "apple IsA edible fruit" and "apple UsedFor eating", while filtering weakly related triples. Conversely, overly strict thresholds may eliminate all retained relations for uncommon objects or specialized concepts. Thus, theta = 0.75 is used as a practical compromise between noisy retrieval and loss of KG grounding.
```

Do not say:

```text
theta = 0.75 is optimal.
```

Say:

```text
theta = 0.75 is a practical retrieval operating point.
```

## 2. Add A Retrieval-Sensitivity Protocol Paragraph In Section 4.3

Current location:

- Section 4.3, `Performance Metrics`.
- The current text explains gold policies and 10 trials.
- Add the theta retrieval protocol after the existing success-rate protocol paragraph and before Section 5.
- Do not present this as another success-rate experiment. Present it as a **retrieval diagnostic**.

### Suggested Text

```text
Threshold retrieval analysis. To examine the robustness of ConceptNet semantic filtering, we perform a retrieval-sensitivity sweep over theta in {0.65, 0.70, 0.75, 0.80, 0.85}. For each threshold, we record retrieval diagnostics rather than using task success as the primary criterion: the fraction of ConceptNet relations retained after filtering, the fraction of OPE objects with no retained relations, and the fraction of URP keywords with no retained relations. These diagnostics directly measure the threshold's effect on KG coverage and are more appropriate for selecting theta than end-to-end success alone, since policy success can also be affected by URP, the planner, and the LLM's ability to compensate without KG evidence.
```

Optional sentence:

```text
Task success is reported separately in the main evaluation, but theta selection is based on the retrieval tradeoff between coverage and noise.
```

Do not add theta success-rate results here. The main performance section already reports task success separately.

## 3. Add A Short Retrieval Subsection In Results

Current location:

- Section 5, after the discussion of Table 2 and before/after the `Ablation Studies` paragraph.
- Keep this subsection focused on retrieval behavior, not success-rate comparison.

Suggested subsection title:

```text
Sensitivity of ConceptNet Retrieval to theta
```

### Suggested Text

```text
We further analyze how the semantic-filtering threshold affects KG evidence available to OPE and URP. Figure X shows that theta strongly controls relation coverage. At theta = 0.65 and theta = 0.70, nearly all candidate relations are retained, which maximizes recall but risks injecting weakly related triples into the LLM context. At theta = 0.75, the system still retains approximately 81% of candidate relations while filtering a meaningful portion of low-similarity evidence. In contrast, theta >= 0.80 causes a sharp coverage collapse: the retained-relation ratio falls to approximately 14% at theta = 0.80 and 4% at theta = 0.85.
```

Add this caution:

```text
The zero-relation diagnostics are particularly important for the reviewer's long-tail concern. At theta = 0.80 and theta = 0.85, many object and keyword queries retain no ConceptNet evidence. In such cases, downstream reasoning becomes less KG-grounded and more dependent on the LLM's internal commonsense. Therefore, we retain theta = 0.75 as the default because it avoids the severe zero-relation regime while still reducing noise compared with lower thresholds.
```

Add this explicit caveat:

```text
This analysis does not imply that theta = 0.75 maximizes success for every task category. Rather, it supports theta = 0.75 as a stable retrieval setting for maintaining KG-grounded evidence across categories.
```

## 4. Main Paper Table To Add

Recommended main-paper table:

```text
Table Y. Retrieval sensitivity of ConceptNet semantic filtering to theta.
```

Recommended caption:

```text
Kept ratio is the fraction of candidate ConceptNet relations retained after semantic filtering. OPE zero-object ratio is the fraction of queried objects for which no relation remains after filtering. URP zero-keyword ratio is the fraction of extracted request keywords for which no relation remains after filtering. The table shows that theta = 0.75 preserves broad KG coverage while avoiding the near-unfiltered regime of lower thresholds and the coverage collapse of stricter thresholds.
```

Current diagnostic values from `threshold_sweep_combined.csv`:

| theta | Kept ratio | OPE zero-object ratio | URP zero-keyword ratio |
| --- | ---: | ---: | ---: |
| 0.65 | 1.000 | 0.276 | 0.077 |
| 0.70 | 0.999 | 0.276 | 0.077 |
| 0.75 | 0.812 | 0.352 | 0.077 |
| 0.80 | 0.142 | 0.812 | 0.432 |
| 0.85 | 0.040 | 0.955 | 0.958 |

Important interpretation:

- `0.65` and `0.70` keep almost everything, so they do not filter much noise.
- `0.80` and `0.85` remove too much evidence, producing many zero-relation objects and keywords.
- `0.75` is the intermediate point: it filters more than `0.65/0.70` while avoiding the severe evidence collapse of `0.80/0.85`.

```text
(TO INSERT: if final values change after regenerating the sweep, replace this table.)
```

## 5. Supplementary Per-Category Table

Use a supplementary table to show the long-tail/coverage effect by category.

Recommended supplementary table:

```text
Table S.X. Per-category OPE zero-object ratio across theta values.
```

Current diagnostic values:

| Category | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
| --- | ---: | ---: | ---: | ---: | ---: |
| explicit ambiguous | 0.24 | 0.24 | 0.24 | 0.76 | 1.00 |
| explicit unambiguous | 0.24 | 0.24 | 0.24 | 0.76 | 1.00 |
| implicit | 0.21 | 0.21 | 0.21 | 0.76 | 1.00 |
| materials | 0.62 | 0.62 | 0.62 | 0.65 | 0.73 |
| risk aware | 0.00 | 0.00 | 0.46 | 1.00 | 1.00 |
| toxicity | 0.34 | 0.34 | 0.34 | 0.94 | 1.00 |

```text
(TO INSERT: replace with final regenerated values if the sweep is rerun.)
```

This table is useful because it directly addresses the reviewer concern about long-tail objects losing useful relations at high thresholds.

## 6. Figures To Add

Use existing figures:

- `scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png`
- `scripts/experiments/theta/results/threshold_sweep_combined_zero_relations.png`

Recommended placement:

- Main paper: include `threshold_sweep_retrieval_v1.png`.
- Supplementary: include per-category plots.

Suggested caption:

```text
Figure X. Effect of theta on ConceptNet retrieval coverage. The plot reports the retained-relation ratio, the OPE zero-object ratio, and the URP zero-keyword ratio. Stricter thresholds sharply increase the fraction of objects and request keywords with no retained KG evidence, especially at theta >= 0.80. The default theta = 0.75 avoids this coverage collapse while still filtering low-similarity relations.
```

The newly generated retrieval-only figure uses a single plot with three curves and the legend outside the plotting area:

- retained-relation ratio;
- OPE zero-object ratio;
- URP zero-keyword ratio.

```text
(DONE: scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png)
```

## 7. Remove Or De-Emphasize Success-Rate Sweep From The Main Paper

Do not include a main-paper table where success rate is the primary evidence for theta selection.

Reason:

- End-to-end success is not a clean measure of threshold quality.
- Explicit tasks may succeed even with little KG evidence.
- A high threshold can look good in task success while actually eliminating KG grounding.
- Reviewer concern is about semantic filtering and relation retention, so retrieval diagnostics are the correct evidence.

Acceptable wording:

```text
We use retrieval diagnostics, rather than end-to-end policy success alone, to select theta because policy success can be confounded by LLM compensation and task explicitness.
```

Do not include the theta success-rate sweep in the supplementary material either. This keeps the response focused: the reviewer questioned the semantic filtering threshold, so the evidence should measure relation filtering directly.

## 8. Add A Limitation/Future Work Sentence

Current location:

- Section 6, `Conclusion and Limitations`.
- The current paper already discusses incomplete ConceptNet coverage.

Add:

```text
The threshold analysis also shows that a fixed global theta cannot optimally serve every object category and task context. Long-tail objects and highly specific concepts may require a lower threshold or top-k fallback to avoid discarding all relevant relations, while common objects may benefit from stricter filtering. Future versions of ConceptBot will investigate adaptive retrieval, for example lowering theta when no relations are retained or retaining the top-k most task-relevant relations under a minimum similarity bound.
```

Optional stronger future-work sentence:

```text
An adaptive retrieval policy could replace the fixed threshold with a hybrid rule: retain relations above theta = 0.75 when available, otherwise back off to the top-k most similar relations down to a lower bound such as theta = 0.65.
```

## 9. Response Letter Text

Use this in the reviewer response:

```text
We thank the reviewer for pointing out that a fixed semantic-filtering threshold may not generalize across categories and long-tail entities. In the revised manuscript, we no longer present theta = 0.75 as an empirically chosen task-performance optimum. Instead, we added a retrieval-sensitivity analysis over theta in {0.65, 0.70, 0.75, 0.80, 0.85}. The analysis reports retained-relation ratio, OPE zero-object ratio, and URP zero-keyword ratio. These diagnostics directly measure the KG coverage/noise tradeoff induced by theta. The results show that low thresholds retain nearly all relations, while stricter thresholds sharply increase zero-relation cases; theta = 0.75 preserves substantial KG coverage while filtering low-similarity evidence. We also added a limitation noting that future work should replace the fixed global threshold with adaptive retrieval or top-k fallback for long-tail concepts.
```

Add final locations after manuscript revision:

```text
(TO INSERT: Section 3.1, Section 4.3, Section 5.X, Figure X, Table Y, and Appendix Table S.X.)
```

## 10. What Still Needs To Be Generated Before Final Manuscript

Required:

1. `(TO INSERT)` Assign final figure/table numbers.
2. `(TO INSERT)` Update the response letter with exact manuscript locations.
3. `(TO INSERT)` If rerunning retrieval extraction changes diagnostics, replace the table values.
4. `(TO INSERT)` Generate qualitative per-relation examples if you want to include the optional qualitative paragraph.

Optional but recommended:

1. Add a top-k fallback experiment:

```text
theta = 0.75; if zero relations are retained, keep top-k relations down to theta = 0.65.
```

2. Run a small qualitative example showing:

- low theta includes noisy triples;
- high theta removes all relations for a complex object;
- theta = 0.75 preserves enough useful relations.

This qualitative example would make the response more intuitive and directly address the reviewer's long-tail concern.

## 11. LaTeX Blocks To Copy Into The Manuscript

### Section 3.1: Semantic Filtering

```latex
\paragraph{Semantic filtering.}
Relevance is measured via cosine similarity, and relations exceeding a threshold $\theta$ are retained:
\[
R_{\mathrm{fil}} = \{r \in R_i : \mathrm{sim}(\mathbf{v}_r,\mathbf{v}_{\mathrm{prop}}) > \theta \}.
\]
Since no global threshold can be universally optimal for all ConceptNet entities, we analyze $\theta$ as a retrieval parameter rather than as a direct task-performance hyperparameter. Specifically, $\theta$ controls the tradeoff between relation coverage and semantic noise. Lower values retain broad commonsense coverage but include weakly related triples, whereas higher values suppress noise but can remove all KG evidence for long-tail objects or task-specific concepts.

We select $\theta = 0.75$ as the default retrieval operating point because it preserves substantial ConceptNet coverage while filtering a meaningful fraction of low-similarity relations. In our threshold sweep over $\theta \in \{0.65, 0.70, 0.75, 0.80, 0.85\}$, $\theta = 0.75$ retains approximately 81\% of candidate relations on average, while stricter thresholds produce a severe coverage collapse: at $\theta = 0.80$ the retained-relation ratio drops to approximately 14\%, and at $\theta = 0.85$ to approximately 4\%. This collapse also increases zero-relation failures in both OPE objects and URP keywords, forcing the downstream LLM to rely more heavily on its internal prior knowledge rather than KG-grounded evidence.
```

### Section 4.3: Retrieval Diagnostic Protocol

```latex
\paragraph{Threshold retrieval analysis.}
To examine the robustness of ConceptNet semantic filtering, we perform a retrieval-sensitivity sweep over $\theta \in \{0.65, 0.70, 0.75, 0.80, 0.85\}$. For each threshold, we record retrieval diagnostics rather than using task success as the primary criterion: the fraction of ConceptNet relations retained after filtering, the fraction of OPE objects with no retained relations, and the fraction of URP keywords with no retained relations. These diagnostics directly measure the threshold's effect on KG coverage and are more appropriate for selecting $\theta$ than end-to-end success alone, since policy success can also be affected by URP, the planner, and the LLM's ability to compensate without KG evidence.
```

### Section 5: Retrieval Results Paragraph

```latex
\subsection{Sensitivity of ConceptNet Retrieval to $\theta$}
We further analyze how the semantic-filtering threshold affects KG evidence available to OPE and URP. Figure~\ref{fig:theta_retrieval} shows that $\theta$ strongly controls relation coverage. At $\theta = 0.65$ and $\theta = 0.70$, nearly all candidate relations are retained, which maximizes recall but risks injecting weakly related triples into the LLM context. At $\theta = 0.75$, the system still retains approximately 81\% of candidate relations while filtering a meaningful portion of low-similarity evidence. In contrast, $\theta \geq 0.80$ causes a sharp coverage collapse: the retained-relation ratio falls to approximately 14\% at $\theta = 0.80$ and 4\% at $\theta = 0.85$.

The zero-relation diagnostics are particularly important for long-tail objects and complex concepts. At $\theta = 0.80$ and $\theta = 0.85$, many object and keyword queries retain no ConceptNet evidence. In such cases, downstream reasoning becomes less KG-grounded and more dependent on the LLM's internal commonsense. Therefore, we retain $\theta = 0.75$ as the default retrieval operating point because it avoids the severe zero-relation regime while still reducing noise compared with lower thresholds. This analysis does not imply that $\theta = 0.75$ maximizes success for every task category; rather, it supports $\theta = 0.75$ as a stable retrieval setting for maintaining KG-grounded evidence across categories.
```

### Main Paper Figure

Use:

```text
scripts/experiments/theta/results/threshold_sweep_retrieval_v1.png
```

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{threshold_sweep_retrieval_v1.png}
    \caption{Effect of the ConceptNet semantic-filtering threshold $\theta$ on retrieval coverage. The plot reports retained-relation ratio, OPE zero-object ratio, and URP zero-keyword ratio. Stricter thresholds sharply reduce KG coverage, especially for $\theta \geq 0.80$. The default $\theta = 0.75$ avoids this coverage collapse while still filtering low-similarity relations.}
    \label{fig:theta_retrieval}
\end{figure}
```

### Main Paper Table

```latex
\begin{table}[t]
\centering
\caption{Retrieval sensitivity of ConceptNet semantic filtering to $\theta$. Kept ratio is the fraction of candidate ConceptNet relations retained after filtering. OPE zero-object ratio is the fraction of queried objects for which no relation remains. URP zero-keyword ratio is the fraction of extracted request keywords for which no relation remains.}
\label{tab:theta_retrieval}
\begin{tabular}{cccc}
\hline
$\theta$ & Kept ratio & OPE zero-object & URP zero-keyword \\
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

### Supplementary Per-Category Table

```latex
\begin{table}[t]
\centering
\caption{Per-category OPE zero-object ratio across semantic-filtering thresholds. Higher values indicate that more queried objects retain no ConceptNet relation after filtering.}
\label{tab:theta_zero_by_category}
\begin{tabular}{lccccc}
\hline
Category & 0.65 & 0.70 & 0.75 & 0.80 & 0.85 \\
\hline
Explicit ambiguous & 0.24 & 0.24 & 0.24 & 0.76 & 1.00 \\
Explicit unambiguous & 0.24 & 0.24 & 0.24 & 0.76 & 1.00 \\
Implicit & 0.21 & 0.21 & 0.21 & 0.76 & 1.00 \\
Materials & 0.62 & 0.62 & 0.62 & 0.65 & 0.73 \\
Risk-aware & 0.00 & 0.00 & 0.46 & 1.00 & 1.00 \\
Toxicity & 0.34 & 0.34 & 0.34 & 0.94 & 1.00 \\
\hline
\end{tabular}
\end{table}
```

### Section 6: Limitation/Future Work

```latex
The threshold analysis also shows that a fixed global $\theta$ cannot optimally serve every object category and task context. Long-tail objects and highly specific concepts may require a lower threshold or a top-$k$ fallback to avoid discarding all relevant relations, while common objects may benefit from stricter filtering. Future versions of ConceptBot will investigate adaptive retrieval, for example lowering $\theta$ when no relations are retained or retaining the top-$k$ most task-relevant relations under a minimum similarity bound.
```
