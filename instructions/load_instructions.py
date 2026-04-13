import json
import sys
from pathlib import Path
from typing import Dict, List, Any

INSTRUCTIONS_DIR = Path(__file__).parent


def load_category(name: str) -> Dict[str, Any]:
    """Load a single category JSON by filename (without .json)."""
    path = INSTRUCTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Category file not found: {path}")
    return json.loads(path.read_text())


def resolve_items(category: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return items with concrete object lists.
    If shared_objects is set, it is copied into each item with objects=None.
    """
    shared = category.get("shared_objects")
    items = []
    for item in category.get("items", []):
        resolved = dict(item)
        if resolved.get("objects") is None and shared is not None:
            resolved["objects"] = list(shared)
        items.append(resolved)
    return items


def list_categories() -> List[str]:
    return sorted([p.stem for p in INSTRUCTIONS_DIR.glob("*.json")])


if __name__ == "__main__":
    # Usage: python instructions/load_instructions.py <category>
    # If no category is provided, list available categories.
    if len(sys.argv) < 2:
        print("Available categories:")
        for name in list_categories():
            print(f"- {name}")
        sys.exit(0)

    name = sys.argv[1]
    cat = load_category(name)
    items = resolve_items(cat)
    print(f"Loaded {len(items)} items from {cat['category']}")
    for item in items:
        print(item)
