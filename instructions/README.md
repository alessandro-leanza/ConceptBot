# Instructions Dataset

This folder contains task instructions and object lists extracted from the paper appendices.
All object names are normalized to lowercase for easier matching in code.

## Schema
Each category file is a JSON object with:
- `category`: category name
- `policy`: category-level policy metadata
  - `order_matters`: whether action order matters
  - `type`: `sequence`, `rules`, or `mixed`
- `shared_objects`: array or null. If not null, the objects apply to all items in the file.
- `items`: list of items
  - `id`: stable identifier
  - `instruction`: string
  - `objects`: array or null (null when `shared_objects` is used)
  - `gold`: gold policy definitions
    - `sequences`: list of alternative action sequences
    - `rules`: mapping of destination -> list of required objects

## Loader Example
Use `load_instructions.py` to load and resolve items:

```bash
# List available categories
python instructions/load_instructions.py

# Print all items in a category
python instructions/load_instructions.py explicit_unambiguous
```

## Validation
Validate instruction files:

```bash
python scripts/experiments/validate_instructions.py
```
