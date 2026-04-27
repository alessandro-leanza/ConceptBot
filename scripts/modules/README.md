# Modules Overview

This folder contains the core ConceptBot modules. The pipeline is organized around **OPE** (Object Properties Extraction), **URP** (User Request Processing), and **Planner** modules, plus utilities for perception and execution.

## Category Pipeline

- `pipeline_config.py`: Central category registry. `CATEGORY_PIPELINE` maps each instruction category to OPE mode, URP mode, target properties, prompt type, cache labels, and dynamic-property policy.

Current category combinations:

- `explicit_unambiguous`, `explicit_ambiguous`, `implicit`: Standard OPE + standard URP.

- `toxicity`: Toxicity-tuned binary OPE + toxicity URP. Target properties include dangerous, safe, poisonous, toxic, venomous, and hazardous.

- `materials`: Materials OPE + materials URP. Target material categories include metal, aluminum, plastic, glass, wood, ceramic, fabric, wax, paper, cardboard, and mixed material.

- `risk_aware`: Risk OPE + risk URP. Risk OPE keeps the 1-5 `Dangerous` score and `DangerousWith` interaction-risk schema.

## OPE (Object Properties Extraction)
OPE enriches detected objects with commonsense properties using ConceptNet relations and LLM reasoning.

- `ope.py`  
  Unified OPE engine. Supports standard binary properties, toxicity-specific binary properties, material extraction, and risk-aware scoring through `mode` / `pipeline_config`. Existing direct `OPE(...)` calls still default to standard binary extraction.

- `ope_score.py`  
  **Scored properties** (1–3): fragile, stable, deformable, toxic, dangerous.

- `ope_mat.py`  
  Legacy/compatibility material classification module. The unified materials path is now exposed through `ope.py` and uses the expanded material set from `pipeline_config.py`.

- `ope_score_par.py`  
  Risk Index implementation (1–5) with interaction risk (`DangerousWith`). Risk mode in `ope.py` delegates here to preserve the existing behavior. Optional Wikipedia/OpenIE fallback for missing ConceptNet info is disabled by default.

## URP (User Request Processing)
URP rewrites free-form user instructions into structured, robot-executable requests using ConceptNet context.

- `urp.py`  
  Unified URP engine. Supports standard, materials, toxicity, and risk prompt modes through `mode` / `pipeline_config`. Standard mode uses object properties without describing them as 1-5 risk scores.

- `urp_risk.py`  
  Legacy/compatibility risk-aware URP. The unified risk prompt mode is now available through `urp.py`.

## Planners
Planner modules generate pick-and-place policies from a structured request.

- `pl_toplog.py`  
  LLM logprob scoring + simple affordance filtering.

- `pl_toplog_prop.py`  
  Like `pl_toplog.py`, but injects object properties into the prompt.

- `pl_posneg.py`  
  Positive/negative prompting + optional affordance factors (RPN, bbox size, properties).

- `pl_iter.py`  
  Iterative LLM selection across multiple candidate actions.

## Perception / Execution

- `kg_yolo.py`  
  YOLO + RealSense + ROS pipeline for real-world object detection.

- `pick_and_place.py`  
  FrankaPy-based execution utilities (pick and place primitives).

## Backend

- `conceptnet_backend.py`  
  ConceptNet access via Hugging Face Space + persistent cache. Used by OPE/URP modules.
