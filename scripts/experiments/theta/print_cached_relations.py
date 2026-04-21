import argparse
import json
from pathlib import Path

CACHE_PATH = Path("cache/conceptnet_cache.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("word", help="term to inspect")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--relations", nargs="*", default=None)
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        print("Cache file not found:", CACHE_PATH)
        return

    cache = json.loads(CACHE_PATH.read_text())
    word = args.word.lower().strip()

    keys = [k for k in cache.keys() if k.startswith(f"{args.lang}:{word}:")]
    if args.relations:
        rel_key = ",".join(sorted(args.relations))
        keys = [k for k in keys if k.endswith(rel_key)]

    if not keys:
        print("No cache entries found for:", word)
        return

    for key in keys:
        print("\nKEY:", key)
        triples = cache[key]
        print("Triples:")
        for t in triples:
            print(f"  {t[0]} {t[1]} {t[2]}")
        print(f"Total: {len(triples)}")


if __name__ == "__main__":
    main()
