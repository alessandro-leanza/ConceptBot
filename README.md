# ConceptBot

ConceptBot is a modular LLM-based robot planning framework that grounds commonsense knowledge in a knowledge graph to resolve underspecified requests and produce feasible, risk-aware pick-and-place plans. The system is organized into three main modules (OPE, URP, Planner), with category-specific OPE/URP behavior configured through a central pipeline registry.

This repository contains the core research code used to prototype the modules and run example pipelines, plus a simulation notebook for PyBullet experiments.

**What ConceptBot Does**
- Grounds detected objects with ConceptNet relations to infer properties and risks.
- Disambiguates user requests into structured, robot-executable commands.
- Plans pick-and-place action sequences with LLM scoring and affordance checks.

**Architecture (Paper Summary)**
- OPE (Object Properties Extraction): retrieves ConceptNet relations for scene objects, filters them by embedding similarity, and asks an LLM to label properties (fragile, dangerous, etc.).
- URP (User Request Processing): extracts keywords from the request, retrieves and filters ConceptNet relations, and rewrites the instruction into a structured, robot-ready query.
- Planner: selects the next action using LLM scoring and combines it with affordance scoring to ensure feasibility.
- Risk Index: optional safety-focused OPE variant that scores object risk (1-5) and interaction risk with other objects.

**Current Pipeline Architecture**
- `scripts/modules/pipeline_config.py` defines `CATEGORY_PIPELINE`, the central mapping from instruction category to OPE mode, URP mode, target properties, prompt type, and cache labels.
- `scripts/modules/ope.py` is the public OPE engine. It supports standard binary properties, toxicity-specific binary properties, material extraction, and risk-aware scoring. Older variant files remain available for compatibility/internal delegation.
- `scripts/modules/urp.py` is the public URP engine. It supports standard, materials, toxicity, and risk-aware prompt modes.
- The theta sweep uses this category mapping:
  - `explicit_unambiguous`, `explicit_ambiguous`, `implicit`: standard OPE + standard URP.
  - `toxicity`: toxicity-tuned OPE + toxicity URP.
  - `materials`: materials OPE + materials URP.
  - `risk_aware`: risk OPE + risk URP.

**Repository Structure**
- `scripts/ConceptBot_Main.py`: main entry point to run the pipeline by toggling modules.
- `scripts/Simulation_Environment.ipynb`: PyBullet-based simulation and evaluation notebook.
- `scripts/modules/pipeline_config.py`: category-specific OPE/URP modes, target properties, prompt types, and cache labels.
- `scripts/modules/ope.py`: unified OPE engine for standard, toxicity, materials, and risk modes.
- `scripts/modules/ope_mat.py`: legacy/compatibility OPE for material classification.
- `scripts/modules/ope_score.py`: OPE with 1-3 property scores.
- `scripts/modules/ope_score_par.py`: legacy/internal OPE Risk Index (1-5) and optional Wikipedia fallback.
- `scripts/modules/urp.py`: unified URP engine for standard, materials, toxicity, and risk modes.
- `scripts/modules/urp_risk.py`: legacy/compatibility URP variant that uses Risk Index outputs.
- `scripts/modules/pl_toplog.py`: Planner using top-logprob scoring + affordances.
- `scripts/modules/pl_toplog_prop.py`: Planner with object properties injected.
- `scripts/modules/pl_posneg.py`: Planner with positive/negative prompting and affordance terms.
- `scripts/modules/pl_iter.py`: Planner using iterative LLM scoring.
- `scripts/modules/kg_yolo.py`: YOLO + RealSense + ROS object detection (real-world).
- `scripts/modules/pick_and_place.py`: FrankaPy execution helpers.

**Setup**
This repo does not include a pinned `requirements.txt`. Install the dependencies you need for your chosen modules. Typical packages include:

```bash
pip install openai requests numpy tiktoken matplotlib scikit-learn wikipedia wikipedia-api openie
```

Optional system dependencies may be required for:
- PyBullet simulation in `scripts/Simulation_Environment.ipynb`
- YOLO, ROS, RealSense for `scripts/modules/kg_yolo.py`
- FrankaPy for real-robot execution

**Configuration**
- Set your OpenAI API key in the environment or directly in the modules (several files contain `openai.api_key = ''`).
- ConceptNet is accessed via HTTP at runtime; no local dump is required.
- The optional Wikipedia fallback in `ope_score_par.py` is disabled by default (`use_wiki = False`).

**Run the Main Pipeline**
Edit flags in `scripts/ConceptBot_Main.py` to enable the desired modules, then run:

```bash
python scripts/ConceptBot_Main.py
```

Minimal example configuration inside `scripts/ConceptBot_Main.py`:
- Enable OPE: `use_OPE = True` (or `use_OPE_score_par = True` for Risk Index)
- Enable URP: `use_URP = True` (or `use_URP_risk = True` when using Risk Index)
- Select a planner: `use_toplog = True` or `use_posneg = True`

The script uses a hard-coded `user_query` and a small set of `found_objects`. Modify these to test different scenarios.

**Run Theta Experiments With Docker**
The experiment Docker Compose file is `docker-compose.experiments.yml`; its service name is `conceptbot-exp`.

List services:

```bash
docker compose -f docker-compose.experiments.yml config --services
```

Run the explicit-unambiguous theta sweep with one trial:

```bash
docker compose -f docker-compose.experiments.yml run --rm conceptbot-exp \
  bash -lc 'PYTHONPATH=. python3 scripts/experiments/theta/threshold_sweep.py \
    --categories explicit_unambiguous \
    --num-trials 1 \
    --out scripts/experiments/theta/results/threshold_sweep_explicit_unambiguous \
    --save-policies \
    --plot'
```

For `explicit_unambiguous`, the sweep uses standard OPE, standard URP, `use_OPE=True`, and the default direct planner. Category-specific OPE/URP combinations are selected through `CATEGORY_PIPELINE` in `scripts/modules/pipeline_config.py`.

**Run the Simulation Notebook**
Open `scripts/Simulation_Environment.ipynb` in Jupyter and follow the cells. It includes:
- PyBullet environment setup with UR5e + Robotiq gripper
- ViLD object detection and CLIPort pick-and-place heatmaps
- LLM-based planning calls

**Paper-to-Code Mapping**
- OPE module: `scripts/modules/ope.py` with category settings in `scripts/modules/pipeline_config.py`; legacy variants remain in `scripts/modules/ope_score.py`, `scripts/modules/ope_mat.py`, and `scripts/modules/ope_score_par.py`.
- Risk Index: risk mode in `scripts/modules/ope.py` delegates to `scripts/modules/ope_score_par.py`; risk URP behavior is available through `scripts/modules/urp.py`.
- URP module: `scripts/modules/urp.py` with standard/materials/toxicity/risk prompt modes.
- Planner (LLM scoring + affordance): `scripts/modules/pl_toplog.py`, `scripts/modules/pl_toplog_prop.py`, `scripts/modules/pl_posneg.py`, `scripts/modules/pl_iter.py`
- Object detection: `scripts/modules/kg_yolo.py` (real-world), notebook uses ViLD for simulation
- Execution on robot: `scripts/modules/pick_and_place.py`

**Known Gaps vs. Paper**
- The prompt lists, task suites, and evaluation harness from the paper are not present in this repo.
- The reported evaluation metrics and benchmarking pipeline are not implemented as standalone scripts here.
- Some affordance components described in the paper (RPN scores, bounding-box size checks) are stubbed or simplified in code, and are not wired to a full perception stack.
- The Wikipedia fallback in OPE exists but is disabled by default and requires extra dependencies (OpenIE).
- No pinned dependency file is provided, so reproducibility requires manual setup.

**License**
See `LICENSE`.
