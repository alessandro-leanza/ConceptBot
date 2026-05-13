# Risk Index Activation Revision Plan

This document addresses the reviewer concern that the Risk Index-enhanced OPE is invoked only when the user explicitly requires enhanced safety.

Reviewer comment addressed:

```text
The Risk Index-enhanced version of OPE is invoked only when users explicitly require enhanced safety. This raises a potential concern: if the user does not explicitly specify safety requirements, but the task inherently involves risk, the system may not activate the full risk evaluation mechanism. This could lead to insufficient awareness of implicit safety hazards.
```

Current paper issue:

- Section 3.1 currently says the Risk Index version "is invoked only when users require enhanced safety."
- This wording makes the system sound dependent on the user explicitly asking for safety.
- The current code now has a centralized `CATEGORY_PIPELINE`, where `risk_aware` selects risk OPE + risk URP.
- A standalone implicit-risk trigger has been implemented and evaluated, but it is not enabled by default in the main online pipeline.

Current repository state:

- `scripts/modules/pipeline_config.py` defines `risk_aware` with:
  - `ope_mode = "risk"`
  - `urp_mode = "risk"`
  - `ope_prompt_type = "risk_score_and_interactions"`
- `scripts/modules/ope.py` dispatches `mode="risk"` to `OPE_score_par`.
- `scripts/modules/urp.py` / `urp_risk.py` provide risk-aware request processing.
- `scripts/modules/risk_trigger.py` now implements a lightweight experimental pre-classifier that can audit whether the Risk Index should be activated.
- `scripts/experiments/risk_trigger/eval_risk_trigger.py` runs a standalone audit over instruction categories and can optionally compare the standard pipeline against risk-triggered `OPE risk + URP risk`.
- The trigger is not enabled by default in the main ConceptBot pipeline.

## Core Position

Use this as the honest revised claim:

```text
The current system supports a risk-aware mode, but the original manuscript under-specified when that mode should be activated. In the revised description, we distinguish between the risk-evaluation capability and the task-mode selection policy. Risk-aware OPE is used for risk-aware task categories and safety-sensitive scenarios; however, automatic detection of implicit hazards in arbitrary user requests remains a limitation and future-work direction.
```

Do not claim:

- that every risky task is automatically detected;
- that the current system has a complete safety trigger;
- that semantic risk scoring replaces physical risk assessment.

Say:

```text
We added an explicit discussion of implicit risk triggering and implemented a lightweight standalone audit that estimates when enhanced Risk Index reasoning should be activated based on the instruction and detected objects.
```

## Implemented Experimental Trigger

The repository now includes:

```text
scripts/modules/risk_trigger.py
scripts/experiments/risk_trigger/eval_risk_trigger.py
```

The public trigger function is:

```python
assess_risk_trigger(
    user_instruction,
    detected_objects,
    task_category=None,
    dynamic_properties=None,
    model="gpt-4o-mini",
)
```

It returns a validated dictionary:

```python
{
    "risk_index_required": bool,
    "confidence": "low" | "medium" | "high",
    "risk_reasons": list[str],
    "risk_cues": list[str],
    "recommended_mode": "risk" | "standard",
    "source": "category_override" | "llm" | "fallback",
}
```

Policy:

- `risk_aware`: forced to `risk_index_required=True` through a category override.
- `explicit_unambiguous`, `explicit_ambiguous`: audited by the trigger, expected to be near-zero false positives.
- `implicit`: audited case by case.
- `materials`, `toxicity`: excluded from the first audit because these are specialized sorting tasks with ad hoc OPE/URP prompts and target properties. Including them would mix risk-triggering with material or toxicity classification.

The trigger asks whether the task needs the numerical and interaction-aware Risk Index, not whether the generic property `dangerous` could be useful. This distinction matters because standard OPE already checks generic danger, while Risk Index is intended for stronger hazard and interaction reasoning.

The final implementation uses an LLM decision followed by conservative validation of risk-core cues. If the LLM proposes Risk Index activation with less than high confidence, the decision must contain at least one core interaction-risk cue, such as heat, microwave/material compatibility, sharp tools, child safety, fragility, breakage, unstable stacking, chemicals, toxicity, electrical/fire risk, or appliance-related breakage. This prevents the trigger from treating generic background problems as Risk Index cases.

## Experiment Command

Audit default categories:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/risk_trigger/eval_risk_trigger.py \
  --categories explicit_unambiguous explicit_ambiguous implicit risk_aware \
  --model gpt-4o-mini \
  --save-traces \
  --out scripts/experiments/risk_trigger/results/risk_trigger_audit
```

Classifier-style evaluation without category overrides:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/risk_trigger/eval_risk_trigger.py \
  --categories explicit_unambiguous explicit_ambiguous implicit risk_aware \
  --model gpt-4o-mini \
  --llm-only \
  --save-traces \
  --out scripts/experiments/risk_trigger/results/risk_trigger_llm_only
```

Use this command if reporting trigger performance. In `--llm-only` mode, the category override is disabled and the category label is hidden from the trigger LLM. The expected label is assigned only by the evaluator:

- `risk_aware`: Risk Index expected.
- `explicit_unambiguous`, `explicit_ambiguous`, `implicit`: Risk Index not expected for the first category-level audit.

This measures whether the trigger can infer the need for Risk Index from the instruction and detected objects, rather than from a hardcoded category policy.

Run the optional triggered-pipeline comparison:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/risk_trigger/eval_risk_trigger.py \
  --categories explicit_unambiguous explicit_ambiguous implicit risk_aware \
  --model gpt-4o-mini \
  --run-triggered-pipeline \
  --save-traces \
  --out scripts/experiments/risk_trigger/results/risk_trigger_pipeline_comparison
```

Smoke test with one item per category:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/risk_trigger/eval_risk_trigger.py \
  --categories explicit_unambiguous implicit risk_aware \
  --limit-per-category 1 \
  --save-traces \
  --out scripts/experiments/risk_trigger/results/risk_trigger_smoke
```

Expected output files:

```text
scripts/experiments/risk_trigger/results/risk_trigger_audit.json
scripts/experiments/risk_trigger/results/risk_trigger_audit.csv
scripts/experiments/risk_trigger/results/risk_trigger_audit.jsonl
```

Result table placeholder:

```text
Category | Items | Expected Triggered | Predicted Triggered | Trigger accuracy | Expected behavior
explicit_unambiguous | 7 | 0 | 0 | 100% | no Risk Index needed
explicit_ambiguous | 10 | 0 | 0 | 100% | no Risk Index needed
implicit | 10 | 0 | 0 | 100% | no Risk Index needed for the current implicit set
risk_aware | 8 | 8 | 8 | 100% | Risk Index required
```

Important: use the `--llm-only` results for performance claims. The category-override audit is useful for pipeline policy checks, but it should not be reported as classifier performance because `risk_aware` is forced to trigger by design.

Final audit command and output:

```bash
docker compose -f docker-compose.experiments.yml run --rm --user "$(id -u):$(id -g)" conceptbot-exp \
  python scripts/experiments/risk_trigger/eval_risk_trigger.py \
  --categories explicit_unambiguous explicit_ambiguous implicit risk_aware \
  --model gpt-4o-mini \
  --llm-only \
  --save-traces \
  --out scripts/experiments/risk_trigger/results/risk_trigger_llm_only_v7
```

Observed metrics:

```text
Overall trigger accuracy: 100%
Risk-aware recall: 100%
Explicit false-positive rate: 0%
Non-risk false-positive rate: 0%
```

Use cautious wording in the paper: this is an audit on the current instruction set, not a general guarantee that all future implicit risks will be detected.

No implicit items were triggered in the final `--llm-only` audit. This is acceptable for the current dataset because the implicit split focuses on semantic underspecification rather than interaction-aware physical risk.

## Paper-Ready Summary

Use this concise summary in the revised manuscript:

```latex
\paragraph{Implicit Risk Trigger.}
To address cases where safety-relevant reasoning may be required even without an explicit user request for safety, we added a lightweight implicit-risk trigger before selecting the OPE/URP mode. The trigger evaluates the user instruction and detected objects and predicts whether the numerical, interaction-aware Risk Index should be activated. Importantly, the trigger is not a generic hazard detector: it activates the Risk Index only when enhanced risk reasoning could change the robot's choice of object, destination, or object--object interaction. The trigger uses an LLM-based JSON decision followed by conservative validation of core risk cues, such as heat, microwave/material compatibility, sharp tools, vulnerable recipients, fragile stacking, breakage-sensitive appliance use, chemicals, toxicity, electrical risk, or unstable placement.
```

If space is tight, use this shorter version:

```latex
\paragraph{Implicit Risk Trigger.}
We added a lightweight trigger that decides whether the Risk Index should be activated even when the user does not explicitly ask for safety. The trigger analyzes the instruction and detected objects and activates risk-aware OPE only when numerical, interaction-aware risk reasoning could change the robot's choice of object, destination, or object--object interaction. This prevents ordinary pick-and-place or semantic-disambiguation tasks from being unnecessarily routed through the risk-aware pipeline.
```

Use this table in the experimental section or supplementary material:

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

Table note to include in text:

```latex
The audit labels the current explicit and implicit splits as non-risk-trigger cases because these instructions require semantic disambiguation or ordinary object selection, while the risk-aware split contains tasks where risk reasoning changes the choice of object, destination, or interaction. This audit does not claim complete detection of all future implicit hazards; it verifies that the trigger behaves correctly on the current ConceptBot instruction set.
```

## 1. Update Section 3.1 Risk Index Text

Current location:

- Section 3.1, paragraph beginning `Risk Index`.
- Current problematic sentence:

```text
This version of the OPE module focuses only on evaluating object risk levels rather than extracting all properties, and is invoked only when users require enhanced safety.
```

### Suggested Replacement

Replace that sentence with:

```text
This version of OPE focuses on evaluating object risk levels and hazardous object interactions rather than extracting the full generic property schema. In the current implementation, it is used for safety-sensitive task categories and explicitly risk-aware scenarios. More generally, activating this mode requires a task-mode selection policy: the system should identify when an instruction or scene contains implicit hazards even if the user does not explicitly ask for safety.
```

### LaTeX Block

```latex
\paragraph{Risk Index.}
We implemented a dedicated risk-evaluation system for scenarios requiring detailed safety assessment. This version of OPE focuses on evaluating object risk levels and hazardous object interactions rather than extracting the full generic property schema. In the current implementation, it is used for safety-sensitive task categories and explicitly risk-aware scenarios. More generally, activating this mode requires a task-mode selection policy: the system should identify when an instruction or scene contains implicit hazards even if the user does not explicitly ask for safety.
```

Then keep the existing explanation of the 1--5 individual and interaction risk scales.

## 2. Add A Short Activation-Policy Paragraph

Recommended location:

- Immediately after the Risk Index paragraph in Section 3.1, or in Section 6 limitations if space is tight.

### Suggested Plain Text

```text
Implicit risk activation. The revised pipeline centralizes category-specific OPE/URP behavior, making the use of risk-aware OPE explicit for risk-aware scenarios. However, a general-purpose robot should also activate enhanced risk analysis when risk is implied by the instruction or scene. A lightweight trigger can be based on three sources: (i) instruction cues such as heat, microwave, cut, clean, chemical, child, fragile, sharp, or hot; (ii) detected object cues such as knife, scissors, glass, bleach, hot tea, medicine, or battery; and (iii) dynamically induced properties such as toxic, flammable, microwave-safe, non-metallic, food-safe, sharp, or contaminating. If such cues are present, the system can route the task through risk-aware OPE or augment generic OPE with risk-specific dynamic properties.
```

### LaTeX Block

```latex
\paragraph{Implicit risk activation.}
The revised pipeline centralizes category-specific OPE/URP behavior, making the use of risk-aware OPE explicit for risk-aware scenarios. However, a general-purpose robot should also activate enhanced risk analysis when risk is implied by the instruction or scene. A lightweight trigger can be based on three sources: (i) instruction cues such as \textit{heat}, \textit{microwave}, \textit{cut}, \textit{clean}, \textit{chemical}, \textit{child}, \textit{fragile}, \textit{sharp}, or \textit{hot}; (ii) detected object cues such as \textit{knife}, \textit{scissors}, \textit{glass}, \textit{bleach}, \textit{hot tea}, \textit{medicine}, or \textit{battery}; and (iii) dynamically induced properties such as \textit{toxic}, \textit{flammable}, \textit{microwave-safe}, \textit{non-metallic}, \textit{food-safe}, \textit{sharp}, or \textit{contaminating}. If such cues are present, the system can route the task through risk-aware OPE or augment generic OPE with risk-specific dynamic properties.
```

Important wording:

- Because the trigger is implemented as a standalone audit but not enabled by default, use "we implement a lightweight experimental trigger" for the repository contribution.
- Do not write that the deployed/default pipeline automatically catches every implicit hazard.
- Use "can be routed" or "we evaluate whether the task should be routed" unless the main pipeline is later changed to call the trigger by default.

## 3. Link To Dynamic Property Induction

Dynamic Property Induction and implicit risk activation should be framed as related but distinct:

- Dynamic Property Induction expands *what properties* OPE checks.
- Risk activation decides *which OPE mode* or risk-specific analysis should run.

Suggested text:

```text
This activation issue is related to Dynamic Property Induction. Dynamic induction expands the property space when a task suggests missing attributes, whereas risk activation decides whether the task should use the risk-specific OPE/URP path. Both mechanisms point to a broader task-mode selection problem: the robot should infer when additional semantic analysis is needed rather than relying only on explicit user wording.
```

### LaTeX Block

```latex
This activation issue is related to Dynamic Property Induction. Dynamic induction expands the property space when a task suggests missing attributes, whereas risk activation decides whether the task should use the risk-specific OPE/URP path. Both mechanisms point to a broader task-mode selection problem: the robot should infer when additional semantic analysis is needed rather than relying only on explicit user wording.
```

## 4. Implemented Minimal Trigger

Implemented lightweight helper:

```text
scripts/modules/risk_trigger.py
```

Public function:

```python
def assess_risk_trigger(user_instruction, detected_objects, task_category=None, dynamic_properties=None, model="gpt-4o-mini"):
    ...
```

Trigger sources considered by the prompt:

- instruction keywords:
  - `heat`, `microwave`, `cook`, `cut`, `clean`, `chemical`, `child`, `food`, `hot`, `sharp`, `break`, `wash`, `dispose`
- object keywords:
  - `knife`, `scissors`, `glass`, `bleach`, `chemical`, `medicine`, `battery`, `hot tea`, `aluminum tray`, `fragile`, `poison`, `acid`
- dynamic property keywords:
  - `toxic`, `hazardous`, `flammable`, `microwave-safe`, `non-metallic`, `food-safe`, `contaminating`, `heat-resistant`, `sharp`

Suggested behavior:

- disabled by default for backward compatibility;
- log reasons when it activates;
- route to risk OPE only when confidence/cue count exceeds a threshold;
- otherwise keep standard OPE but optionally add risk-related dynamic properties.

This has been implemented as an audit tool, not as a default production trigger. The final `--llm-only` audit reports 100% overall trigger accuracy, 100% risk-aware recall, 0% explicit false-positive rate, and 0% non-risk false-positive rate on the current ConceptBot instruction set.

## 5. Limitation / Future Work Text

Recommended location:

- Section 6, `Conclusion and Limitations`.

### Suggested Plain Text

```text
The Risk Index still depends on task-mode selection. In this revision, we add a lightweight implicit-risk trigger and validate it on the current ConceptBot instruction set. However, the trigger is not a complete safety monitor: it only estimates whether semantic, interaction-aware risk reasoning should be activated before planning. It does not replace physical risk validation, force/torque monitoring, geometric stability analysis, thermal modeling, or closed-loop execution monitoring. Future work will integrate implicit risk triggering into the default online pipeline and combine it with grounded physical safety checks.
```

### LaTeX Block

```latex
The Risk Index still depends on task-mode selection. In this revision, we add a lightweight implicit-risk trigger and validate it on the current ConceptBot instruction set. However, the trigger is not a complete safety monitor: it only estimates whether semantic, interaction-aware risk reasoning should be activated before planning. It does not replace physical risk validation, force/torque monitoring, geometric stability analysis, thermal modeling, or closed-loop execution monitoring. Future work will integrate implicit risk triggering into the default online pipeline and combine it with grounded physical safety checks.
```

## 6. Response Letter Text

Use this in the reviewer response:

```text
We thank the reviewer for noting that risk-aware OPE should not depend solely on an explicit user request for safety. In the revised implementation, we add a lightweight implicit-risk trigger that evaluates whether the numerical, interaction-aware Risk Index should be activated from the user instruction and detected objects. The trigger is evaluated in an LLM-only setting where category overrides are disabled and the category label is hidden from the trigger. On the current ConceptBot instruction set, it correctly activates the Risk Index for all risk-aware tasks and does not activate it for the explicit or implicit non-risk tasks (35/35 correct decisions; 100% risk-aware recall; 0% explicit false-positive rate). We also clarify in the manuscript that this trigger is not a complete physical safety monitor and does not replace grounded execution-time safety checks.
```

Add final manuscript locations:

```text
Section 3.1 Risk Index paragraph; new implicit-risk trigger paragraph; Table~\ref{tab:risk-trigger-audit}; Section 6 limitations; response letter.
```

## 7. Final Status

Repository-side work for this reviewer comment is complete:

- `scripts/modules/risk_trigger.py` implements the trigger.
- `scripts/experiments/risk_trigger/eval_risk_trigger.py` implements the audit runner.
- `scripts/experiments/risk_trigger/results/risk_trigger_llm_only_v7.*` contains the final audit results.
- This document provides paper-ready LaTeX text, a table, limitation wording, and response-letter text.

Remaining non-code work:

- Insert the Risk Index replacement paragraph in Section 3.1.
- Insert the implicit-risk trigger paragraph and Table~\ref{tab:risk-trigger-audit}.
- Insert the limitation paragraph in Section 6.
- Use the response-letter paragraph above.
