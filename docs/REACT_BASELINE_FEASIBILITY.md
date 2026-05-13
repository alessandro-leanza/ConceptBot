# ReAct Baseline Feasibility

This note evaluates whether the cloned `external_baselines/ReAct` repository can be used as an external baseline on the ConceptBot instruction benchmark.

## What the cloned ReAct repo contains

The cloned repository is a prompt-and-notebook release for the ICLR 2023 ReAct paper. Its main components are:

- notebooks for `HotpotQA`, `FEVER`, `ALFWorld`, and `WebShop`
- prompt JSON files under `prompts/`
- task wrappers in [wrappers.py](/home/alessandro/ConceptBot/external_baselines/ReAct/wrappers.py)
- environment-specific helpers such as [wikienv.py](/home/alessandro/ConceptBot/external_baselines/ReAct/wikienv.py)

It does not provide:

- a reusable robot-planning package
- a generic CLI evaluation harness
- native support for ConceptBot-style actions such as `robot.pick_and_place(object, destination)`

## What is transferable

The following ReAct ideas transfer cleanly to ConceptBot:

- interleaved `Thought -> Action -> Observation` prompting
- stepwise reasoning with external observations between actions
- a strict action vocabulary and a stop action
- iterative correction after invalid actions or weak observations

These are method-level ideas and do not depend on the original environments.

## What is not transferable as-is

Several parts of the original repo are tightly coupled to its benchmark tasks:

- action types such as `Search[...]`, `Lookup[...]`, and `Finish[...]`
- environment assumptions from Wikipedia lookup, ALFWorld, and WebShop
- notebook-driven execution instead of a scriptable benchmark harness
- the older OpenAI completion API style shown in the notebooks

Those choices do not match ConceptBot's benchmark, where the action schema is already fixed and the evaluator expects `robot.pick_and_place(...)` plans.

## Fairness on the ConceptBot benchmark

The most defensible comparison is a harness-first ReAct baseline:

- use ReAct as a prompting paradigm, not as a frozen external codebase
- keep ConceptBot's existing instruction loader and gold evaluator
- remove OPE, URP, and ConceptNet from the baseline
- let the ReAct baseline generate the final action sequence directly

This is fairer than forcing the original repo into a mismatched robot setting because:

- the benchmark inputs stay identical
- the action space stays identical
- the evaluator stays identical
- only the reasoning-and-planning method changes

## Final judgment

Direct reuse of `external_baselines/ReAct` as an executable baseline for ConceptBot is not practical.

Method-style adaptation inside the ConceptBot harness is practical and is the recommended path. The right comparison is:

- `ConceptBot`: OPE + URP + planner
- `ReAct baseline`: direct ReAct prompting from instruction and scene objects to `robot.pick_and_place(...)` actions

This keeps the evaluation controlled while still using a real external baseline family instead of another internal ablation.
