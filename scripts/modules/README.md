# Modules Overview

This folder contains the core ConceptBot modules. The pipeline is organized around **OPE** (Object Properties Extraction), **URP** (User Request Processing), and **Planner** modules, plus utilities for perception and execution.

## OPE (Object Properties Extraction)
OPE enriches detected objects with commonsense properties using ConceptNet relations and LLM reasoning.

- `ope.py`  
  **Binary property extraction** (Yes/No): dangerous, fragile, deformable, hold liquid, safe, stable, poisonous.

- `ope_score.py`  
  **Scored properties** (1–3): fragile, stable, deformable, toxic, dangerous.

- `ope_mat.py`  
  **Material classification**: metal, plastic, glass, wood, ceramic, fabric, wax. Used for material-sorting tasks.

- `ope_score_par.py`  
  **Risk Index variant** (1–5) with interaction risk (`DangerousWith`). Optional Wikipedia/OpenIE fallback for missing ConceptNet info (disabled by default).

## URP (User Request Processing)
URP rewrites free-form user instructions into structured, robot-executable requests using ConceptNet context.

- `urp.py`  
  Standard URP (keywords + object relations + few-shot examples). Returns an action-oriented instruction.

- `urp_risk.py`  
  Risk-aware URP. Same structure as `urp.py`, but consumes Risk Index outputs (from `ope_score_par.py`).

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
