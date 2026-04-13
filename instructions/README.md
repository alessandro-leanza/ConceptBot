# Instructions Dataset

This folder contains task instructions and object lists extracted from the paper appendices.
All object names are normalized to lowercase for easier matching in code.

## Schema
Each category file is a JSON object with:
- `category`: category name
- `shared_objects`: array or null. If not null, the objects apply to all items in the file.
- `items`: list of items
  - `id`: stable identifier
  - `instruction`: string
  - `objects`: array or null (null when `shared_objects` is used)

This gives a uniform structure while avoiding object list duplication for categories that share the same object set.

## Loader Example
Use `load_instructions.py` to load and resolve items:

```bash
# List available categories
python instructions/load_instructions.py

# Print all items in a category
python instructions/load_instructions.py explicit_unambiguous
```

Or import in your code:

```python
from instructions.load_instructions import load_category, resolve_items

cat = load_category("risk_aware")
items = resolve_items(cat)
```
