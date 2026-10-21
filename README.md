# ConceptBot

**Knowledge-Graph–Grounded Commonsense for Task Decomposition in LLM Robot Planning**

[Project Page](https://alessandro-leanza.github.io/projects/conceptbot/) · [Paper](https://arxiv.org/abs/2509.00570) · [Code](https://github.com/alessandro-leanza/ConceptBot)

**Accepted for publication in [Biomimetic Intelligence and Robotics (BIROB)](https://www.sciencedirect.com/journal/biomimetic-intelligence-and-robotics).** The arXiv preprint is available below;
the citation will be updated when the final publication metadata become available.

ConceptBot combines large language model reasoning with ConceptNet commonsense
knowledge for ambiguity-aware and risk-aware task-level robot planning.

<p align="center">
  <img src="https://raw.githubusercontent.com/alessandro-leanza/alessandro-leanza.github.io/main/public/projects/conceptbot/architecture.png" alt="ConceptBot overview: Object Properties Extraction enriches the scene, User Request Processing interprets the instruction, and the Planner generates pick-and-place actions." width="520">
</p>

*OPE → URP → Planner: from commonsense scene enrichment to structured intent and scene-valid actions.*

## Overview

1. **OPE — Object Properties Extraction.** Retrieves and filters ConceptNet
   relations and infers task-relevant semantic properties of detected objects.
2. **URP — User Request Processing.** Uses the user instruction, detected scene,
   object properties, semantic relations, and robot capabilities to resolve
   ambiguous or implicit requests into structured object–destination operations.
3. **Planner.** Grounds the structured request into sequential scene-valid
   pick-and-place actions, ending with `done()`.

The optional semantic **Risk Index** estimates risk for objects and object–object
interactions. It informs task-level interpretation; it is **not a physics-based
safety certificate** or a replacement for geometric, grasping, or control validation.

## Key results

| Evaluation | Reported result |
| --- | --- |
| Implicit tasks | ConceptBot **87.0%**; SayCan 31.0%; ReAct 27.0%; ProgPrompt 13.0%. |
| Risk-aware tasks | ConceptBot **76.2%**; SayCan 15.0%; ReAct 37.5%; ProgPrompt 22.5%. |
| SafeAgentBench | Observed overall BSS **80%**; next-highest reported comparison ReAct + ThinkSafe 46%. |
| Controlled PyBullet/Franka deployments | **45/50** overall controlled task completions; all **45/45** planning-correct policies completed execution. |

The benchmark percentages measure symbolic planning-policy correctness, not physical robot success. Controlled deployment runs are reported separately and demonstrate planner-to-execution integration under supported perception and manipulation conditions.

These are paper results, not results produced by the quick-start demo.

## Quick start

Run commands from the repository root. The container uses Python 3.11; local
installation and credential-free checks were also tested with Python 3.12.

### Local setup

```bash
git clone https://github.com/alessandro-leanza/ConceptBot.git
cd ConceptBot
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env
```

Edit `.env` locally and set `OPENAI_API_KEY`. Do not commit that file. For a local
Bash session, export the values before running the demo:

```bash
set -a
. ./.env
set +a
pip install -r requirements.txt
python scripts/demo.py
```

The local Python entry point does not automatically load `.env`; merely editing
it is insufficient without exporting the variables. Source only your own trusted
`.env` file. Docker Compose injects the file automatically.

The demo runs a small, hard-coded scene through OPE → URP → Planner and prints the
instruction, structured request, and plan. It does not load the full benchmark or
execute a robot. It requires a valid OpenAI key, access to the configured chat and
embedding models, and access to the Hugging Face ConceptNet service
`cstr/conceptnet_normalized` through `gradio_client`. API calls may incur charges.
A fresh checkout does not include populated knowledge or embedding caches.

**Validation scope:** dependency installation, demo imports, the example loader,
and the missing-key failure path were tested. End-to-end LLM generation requires
external credentials and services and has not been validated here.

### Credential-free example inspection

```bash
python instructions/load_instructions.py
python instructions/load_instructions.py examples
```

The first command lists JSON filenames, including `scenes` and `gold`; the second
prints the two legacy examples. This loader is for the category/example schema,
not the split benchmark format: do **not** pass `scenes` or `gold` to it. See the
[benchmark specification](instructions/README.md) for a split-format loading example.

### Docker

Requires Docker Engine and Docker Compose v2. Create and edit `.env` as above,
then run:

```bash
docker compose config --quiet
docker compose build
docker compose run --rm conceptbot
# Run the LLM-backed demo instead of the default instruction listing:
docker compose run --rm conceptbot python scripts/demo.py
```

The default command lists instruction files and needs no API credentials; the
demo has the same external requirements as the local workflow. Compose mounts
the checkout at `/workspace`. The image also contains the source and benchmark
files, so it can be used without that mount:

```bash
docker build -t conceptbot .
docker run --rm conceptbot
docker run --rm --env-file .env conceptbot python scripts/demo.py
```

## Benchmark

The machine-readable benchmark in [`instructions/`](instructions/) contains
**47 instructions**:

| Category | Count |
| --- | ---: |
| Unambiguous | 7 |
| Ambiguous | 10 |
| Implicit | 10 |
| Risk-aware | 8 |
| Materials | 7 |
| Toxicity | 5 |

- [`scenes.json`](instructions/scenes.json) is runtime-facing: instructions and
  symbolic scenes without reference policies.
- [`gold.json`](instructions/gold.json) contains admissible policies and matching
  semantics for **post-generation scoring only**. Gold data must never be used
  by OPE, URP, the Planner, or a baseline during generation.

Evaluation checks membership in expert-validated admissible policy sets rather
than literal equality with one reference sequence. Depending on the item, matching
uses an exact action multiset, an ordered sequence, or an object–destination
assignment. Extra, duplicate, invalid, unsafe, or unrequested actions are not
accepted; causal ordering is enforced where specified.

The paper evaluates each instruction through 10 complete policy-generation
repetitions, with the instruction as the statistical unit. The quick start is a
pipeline demonstration, not a benchmark runner. This checkout provides the
specification and pipeline modules, but no complete benchmark/scoring command.
See [`instructions/README.md`](instructions/README.md) for the full data contract.

## Repository structure

```text
ConceptBot/
├── README.md
├── CITATION.cff
├── LICENSE
├── .env.example
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── scripts/
│   ├── demo.py                      # Minimal OPE → URP → Planner example
│   └── modules/
│       ├── README.md
│       ├── ope.py                   # Object Properties Extraction
│       ├── urp.py                   # User Request Processing
│       ├── pl_vote.py               # Scene-valid action planner
│       ├── conceptnet_backend.py    # ConceptNet retrieval and normalization
│       ├── semantic_cache.py        # Embedding, keyword, and similarity caches
│       ├── pipeline_config.py       # Category-specific OPE/URP configuration
│       ├── ope_score_par.py         # Semantic Risk Index backend
│       ├── risk_trigger.py          # Optional implicit-risk activation helper
│       └── dynamic_properties.py    # Optional Dynamic Property Induction
└── instructions/
    ├── README.md                    # Benchmark specification
    ├── scenes.json                  # 47 runtime-facing items
    ├── gold.json                    # 47 scoring-only annotations
    ├── examples.json                # Two legacy-format examples
    └── load_instructions.py         # Legacy example/category loader
```

See also the [module index](scripts/modules/README.md).

## Optional features

- **Dynamic Property Induction:** utilities in `dynamic_properties.py` can infer
  task-conditioned properties beyond the base schema. Callers must explicitly
  invoke them and pass the result into OPE. It is disabled in every default
  category configuration and is not part of the main benchmark configuration.
- **Semantic Risk Index / implicit risk trigger:** `risk_aware` selects the risk
  OPE/URP configuration. `risk_trigger.py` separately provides optional activation
  logic for implicit risk; the demo does not call it and it is disabled from the
  main pipeline by default. Do not conflate this trigger with the evaluated
  risk-aware configuration.
- **Wikipedia/OpenIE coverage extension:** the risk backend includes an optional
  Wikipedia fallback, disabled by `use_wiki = False`. The requirements include
  the Wikipedia clients, but Stanford OpenIE and its Java/backend setup are
  separate prerequisites. This extension is not enabled in the main benchmark
  configuration and is not required for the quick start.

## Reproducibility and scope

This repository primarily covers the **semantic/task-level planning pipeline**.
Robot-specific perception and manipulation dependencies such as FrankaPy,
RealSense, YOLO, ROS, and lab-specific interfaces are environment-specific and
intentionally not installed by the default `requirements.txt`. Cloning the
repository does not reproduce a physical Franka deployment.

The demo uses the existing model names, thresholds, prompts, and category
configuration. Dependencies are currently unpinned and the base image is a
floating Python tag; this is not an archival environment lock. For experiments,
record the commit, Python/package versions, model availability, configuration,
and cache state. Runtime caches under `cache/` are local generated artifacts.
`CONCEPTNET_CACHE_ONLY=1` disables uncached ConceptNet queries; a cache miss then
returns no relations, so this is not equivalent to reproducing a grounded run.
The demo still needs OpenAI access for uncached model calls.

## Citation

Use the **arXiv/preprint** citation below. [`CITATION.cff`](CITATION.cff) provides
GitHub's “Cite this repository” metadata and identifies the preprint as the
preferred citation. This preprint citation is provisional while the accepted
Biomimetic Intelligence and Robotics paper awaits final publication metadata. No final DOI, volume, issue, or pages
are asserted.

```bibtex
@article{leanza2025conceptbot,
  title={ConceptBot: Knowledge-Graph--Grounded Commonsense for Task Decomposition in LLM Robot Planning},
  author={Leanza, Alessandro and Moroncelli, Angelo and Vizzari, Giuseppe and Braghin, Francesco and Roveda, Loris and Spahiu, Blerina},
  journal={arXiv preprint arXiv:2509.00570},
  year={2025}
}
```

## License

[Apache License 2.0](LICENSE). The existing license is unchanged.
