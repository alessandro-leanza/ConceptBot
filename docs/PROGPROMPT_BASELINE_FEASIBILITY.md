# ProgPrompt Baseline Feasibility

This note evaluates whether the cloned `external_baselines/progprompt-vh` repository can be used as an external baseline on the ConceptBot instruction benchmark.

## What the cloned ProgPrompt repo contains

The cloned repository is the code release for ProgPrompt on the VirtualHome benchmark. Its key components are:

- a VirtualHome-specific evaluation script in [run_eval.py](/home/alessandro/ConceptBot/external_baselines/progprompt-vh/scripts/run_eval.py)
- environment execution helpers in [utils_execute.py](/home/alessandro/ConceptBot/external_baselines/progprompt-vh/scripts/utils_execute.py)
- program-like prompt examples and task data under `data/`
- state checks expressed with `assert(...)`-style lines

It assumes:

- VirtualHome is installed
- the Unity simulator is available
- plans are executed against an environment graph
- evaluation is based on final-state and execution metrics, not only textual action matching

It does not provide:

- a reusable pick-and-place planning package
- native support for `robot.pick_and_place(object, destination)`
- a benchmark harness compatible with ConceptBot instructions and evaluator

## What is transferable

The following ProgPrompt ideas transfer well to ConceptBot:

- programmatic plan generation instead of free-form reasoning
- one statement per line
- explicit action sequences with simple guards
- lightweight state checks such as `assert(...)`

These elements can be adapted without bringing over the VirtualHome stack.

## What is not transferable as-is

Several parts of the original repo are tightly bound to VirtualHome:

- the VirtualHome action vocabulary
- Unity and evolving-graph execution
- object identifiers and room graphs
- environment-based success metrics
- the older OpenAI completion API usage in the released scripts

Those assumptions do not match ConceptBot, where the benchmark expects `robot.pick_and_place(...)` actions and already has a gold-policy evaluator.

## Fairness on the ConceptBot benchmark

The most defensible comparison is a harness-first ProgPrompt baseline:

- use ProgPrompt as a programmatic prompting style, not as a frozen environment-specific implementation
- keep the same instruction files, scene objects, and action schema as ConceptBot
- remove OPE, URP, ConceptNet, and the internal planner from the baseline
- execute the generated program with a deterministic local checker

This is fairer than forcing the VirtualHome code into a mismatched benchmark because:

- the benchmark inputs stay identical
- the action space stays identical
- the evaluator stays identical
- only the planning style changes

## Final judgment

Direct reuse of `external_baselines/progprompt-vh` as an executable baseline for ConceptBot is not practical.

Method-style adaptation inside the ConceptBot harness is practical and is the recommended path. The right comparison is:

- `ConceptBot`: OPE + URP + planner
- `ProgPrompt baseline`: direct program-style planning from instruction and scene objects to `robot.pick_and_place(...)` actions

This preserves the core idea of ProgPrompt while keeping the evaluation controlled and comparable.
