# Public Modules

This directory contains the core public ConceptBot modules:

- `conceptnet_backend.py`: ConceptNet access and relation normalization.
- `semantic_cache.py`: local cache helpers for embeddings, keywords, and LLM calls.
- `pipeline_config.py`: category-specific OPE/URP configuration.
- `dynamic_properties.py`: optional task-conditioned property expansion utilities.
- `risk_trigger.py`: optional risk-aware mode trigger utility.
- `ope.py`: unified Object Properties Extraction module.
- `urp.py`: unified User Request Processing module.
- `pl_vote.py`: deterministic next-action planner used in the public release.
- `ope_score_par.py`: risk-index OPE backend used by `ope.py` in risk mode.
