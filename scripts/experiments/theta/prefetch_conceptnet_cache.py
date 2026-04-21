import argparse
import re
from typing import Iterable, Set

from instructions.load_instructions import load_category, resolve_items
from scripts.modules.conceptnet_backend import get_conceptnet_relations


DEFAULT_CATEGORIES = [
    "explicit_unambiguous",
    "explicit_ambiguous",
    "implicit",
    "risk_aware",
    "materials",
    "toxicity",
]

STOPWORDS = {
    "the", "a", "an", "to", "and", "or", "in", "on", "of", "for", "with", "is", "are",
    "be", "me", "my", "i", "you", "we", "it", "them", "that", "this", "these", "those",
    "into", "by", "at", "from", "as", "not", "do", "does", "did", "so", "but", "if",
    "all", "one", "two", "three", "up", "down", "out", "off", "there", "here",
}
OPE_RELATIONS = sorted(["MadeOf", "UsedFor", "IsA", "HasProperty", "CapableOf", "PartOf", "RelatedTo"])
URP_RELATIONS = sorted(["IsA", "UsedFor", "HasProperty", "CapableOf", "MannerOf"])
RELATION_SETS = {
    "ope": OPE_RELATIONS,
    "urp": URP_RELATIONS,
}


def extract_terms(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _prefetch_terms(terms: Iterable[str], relations: list[str]) -> tuple[int, int]:
    succeeded = 0
    failed = 0
    for term in sorted(terms):
        try:
            _ = get_conceptnet_relations(term, lang="en", relations=relations)
            succeeded += 1
        except Exception:
            failed += 1
    return succeeded, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--modes", nargs="+", choices=sorted(RELATION_SETS.keys()), default=sorted(RELATION_SETS.keys()))
    args = parser.parse_args()

    terms: Set[str] = set()

    for category in args.categories:
        cat = load_category(category)
        items = resolve_items(cat)
        for item in items:
            for obj in item["objects"]:
                terms.add(obj.lower())
            terms.update(extract_terms(item["instruction"]))

    total = len(terms)
    print("Prefetch summary")
    print("Total unique terms:", total)
    for mode in args.modes:
        relations = RELATION_SETS[mode]
        succeeded, failed = _prefetch_terms(terms, relations)
        print(f"Mode: {mode}")
        print("  Relations:", ",".join(relations))
        print("  Succeeded:", succeeded)
        print("  Failed:", failed)


if __name__ == "__main__":
    main()
