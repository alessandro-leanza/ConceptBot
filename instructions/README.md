# Instructions Examples

The public release includes a small example instruction file, `examples.json`, to document the benchmark schema without releasing the full internal benchmark suite.

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
