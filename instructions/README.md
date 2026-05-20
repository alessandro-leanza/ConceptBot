# Instructions Examples

This directory includes `examples.json`, a compact instruction file that documents the benchmark schema used by the loader and demo utilities.

## Schema

Each instruction file is a JSON object with:

- `category`: category name.
- `policy`: category-level metadata.
  - `order_matters`: whether action order matters.
  - `type`: `sequence`, `rules`, or `mixed`.
- `shared_objects`: optional object list shared by all items.
- `items`: instruction records.
  - `id`: stable identifier.
  - `instruction`: natural-language user request.
  - `objects`: item-specific object list, or `null` when `shared_objects` is used.
  - `gold`: example target policy.

## Loader Example

```bash
python instructions/load_instructions.py
python instructions/load_instructions.py examples
```
