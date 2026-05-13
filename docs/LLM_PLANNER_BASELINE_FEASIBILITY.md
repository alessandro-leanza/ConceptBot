# LLM-Planner Baseline Feasibility

This note evaluates whether the cloned `external_baselines/LLM-Planner` repository can be used as an external baseline on the ConceptBot instruction benchmark.

## What the cloned LLM-Planner repo contains

The cloned repository is the code release for LLM-Planner on embodied household planning tasks. Its main components are:

- `hlp/`: a high-level prompt generator with kNN retrieval over a small ALFRED-style task set.
- `e2e/`: an end-to-end ALFRED/AI2-THOR agent using the high-level planner.
- `knn_set.pkl`: a few-shot example dataset used to retrieve similar in-context examples.
- ALFRED actions such as `Navigation`, `PickupObject`, `PutObject`, `OpenObject`, and `ToggleObjectOn`.

The end-to-end implementation assumes:

- the ALFRED dataset is downloaded;
- AI2-THOR is installed and display/X server support is available;
- generated high-level actions are grounded by ALFRED/THOR connectors;
- evaluation uses simulator execution and ALFRED task success.

## What is transferable

The following LLM-Planner ideas transfer to ConceptBot:

- grounded planning from natural-language task descriptions;
- visible-object constraints in the prompt;
- few-shot example retrieval from similar prior tasks;
- the `Completed plans / Visible objects / Next Plans` prompt structure;
- direct generation of a high-level action sequence.

These are method-level ideas and do not require the ALFRED simulator.

## What is not transferable as-is

Direct reuse is not practical because the original repo is tightly coupled to ALFRED:

- the native action vocabulary does not match `robot.pick_and_place(object, destination)`;
- ALFRED/THOR low-level grounding is not available in ConceptBot;
- the simulator success metric does not match ConceptBot's gold-policy evaluator;
- the original kNN examples use ALFRED tasks and objects, not ConceptBot scenes;
- the released high-level planner uses older OpenAI completion-style APIs in `hlp/`.

Running the official end-to-end agent would test an ALFRED planner, not the ConceptBot benchmark.

## Fairness on the ConceptBot benchmark

The defensible comparison is a harness-first LLM-Planner-style baseline:

- keep ConceptBot's instruction files, scene objects, action schema, and gold evaluator;
- remove OPE, URP, ConceptNet, and the internal `DIRECT` planner from the baseline;
- preserve the LLM-Planner prompt style with retrieved few-shot examples;
- ask the model to output `robot.pick_and_place(...)` steps and `done()` directly.

This makes the comparison controlled while still testing a recognizable external planning family.

## Redundancy risk

This baseline can become redundant if it collapses into a plain direct LLM planner. To remain useful, it must preserve the LLM-Planner-specific behavior:

- use task-similarity retrieval for examples;
- include visible objects and valid destinations explicitly;
- format examples and the test prompt as `Task description`, `Completed plans`, `Visible objects are`, and `Next Plans`;
- produce direct high-level plans without OPE/URP.

If results and prompts are indistinguishable from the existing `DIRECT` planner, the baseline should be reported cautiously or left as an internal feasibility check.

## Final judgment

Direct reuse of `external_baselines/LLM-Planner` as an executable baseline for ConceptBot is not practical.

Method-style adaptation inside the ConceptBot harness is practical and is the recommended path. The right comparison is:

- `ConceptBot`: OPE + URP + planner;
- `LLM-Planner-style baseline`: few-shot grounded planning from instruction and visible objects directly to `robot.pick_and_place(...)` actions.

In the paper or rebuttal, this should be described as an adapted LLM-Planner-style external baseline, not as the official ALFRED/THOR implementation.
