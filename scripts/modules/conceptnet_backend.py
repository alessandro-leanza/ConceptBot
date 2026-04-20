import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Any

try:
    from gradio_client import Client
except Exception:  # pragma: no cover
    Client = None

CACHE_PATH = Path("cache/conceptnet_cache.json")
DEFAULT_RELATIONS = [
    "IsA",
    "RelatedTo",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "AtLocation",
    "HasProperty",
    "MadeOf",
    "Synonym",
    "Antonym",
]

_client = None


def _get_client():
    global _client
    if _client is None:
        if Client is None:
            raise ImportError("gradio_client is not installed. Run: pip install gradio_client")
        _client = Client("cstr/conceptnet_normalized")
    return _client


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _make_cache_key(lang: str, word: str, relations: List[str]) -> str:
    rel_key = ",".join(sorted(relations))
    return f"{lang}:{word}:{rel_key}"


def _normalize_word(word: str) -> str:
    return word.strip().lower()


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"`", "", text)
    text = re.sub(r"\\*\\*", "", text)
    text = re.sub(r"\\*", "", text)
    text = re.sub(r"\\[.*?\\]", "", text)
    text = re.sub(r"\\(.*?\\)", "", text)
    text = text.replace("**", "").replace("*", "")
    return text.strip()


def _parse_hf_result(word: str, result: Any, relations: List[str]) -> List[Tuple[str, str, str]]:
    """
    Convert HF response into list of (start, rel, end) triples.
    Handles common formats: list of dicts, dict of relations, or text.
    """
    triples: List[Tuple[str, str, str]] = []
    if isinstance(result, dict):
        # If dict maps relation -> list of targets
        for rel, targets in result.items():
            if rel not in relations:
                continue
            if isinstance(targets, list):
                for t in targets:
                    if isinstance(t, dict):
                        end = t.get("term") or t.get("end") or t.get("label")
                    else:
                        end = str(t)
                    if end:
                        triples.append((word, rel, end))
        return triples

    if isinstance(result, list):
        # If list of dicts with relation/label fields
        for item in result:
            if isinstance(item, dict):
                rel = item.get("relation") or item.get("rel") or item.get("predicate")
                end = item.get("end") or item.get("term") or item.get("label") or item.get("target")
                if rel in relations and end:
                    triples.append((word, rel, str(end)))
            else:
                # If list of strings, try "rel: target" format
                text = str(item)
                for rel in relations:
                    if text.startswith(rel + ":"):
                        end = text.split(":", 1)[1].strip()
                        if end:
                            triples.append((word, rel, end))
        return triples

    # Fallback: parse text lines
    if isinstance(result, str):
        lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
        current_rel = None
        rel_set = set(relations)
        for ln in lines:
            if ln.startswith("## "):
                header = ln[3:].strip()
                current_rel = header if header in rel_set else None
                continue

            if "→" not in ln:
                continue

            # Example line:
            # - **apple** IsA → *fruit* [12.3]
            # - *stem* PartOf → **apple** [1.0]
            line = ln.lstrip("- ").strip()

            rel_in_line = None
            for rel in rel_set:
                if f" {rel} " in line:
                    rel_in_line = rel
                    break
            rel_used = rel_in_line or current_rel
            if rel_used not in rel_set:
                continue

            left, right = line.split("→", 1)
            left = _clean_text(left)
            right = _clean_text(right)

            # Remove relation token from left side if present
            if rel_used in left:
                left = left.replace(rel_used, "").strip()

            start = left
            end = right
            end = end.replace("**", "").replace("*", "").strip()
            start = start.replace("**", "").replace("*", "").strip()
            if start and end:
                triples.append((start, rel_used, end))
        return triples

    return triples


def _normalize_triples(triples: List[Any]) -> List[Tuple[str, str, str]]:
    norm = []
    for t in triples:
        if isinstance(t, tuple):
            norm.append(t)
        elif isinstance(t, list) and len(t) >= 3:
            norm.append((str(t[0]), str(t[1]), str(t[2])))
    return norm


def get_conceptnet_relations(
    word: str,
    lang: str = "en",
    relations: Optional[List[str]] = None,
    cache_only: Optional[bool] = None,
) -> List[Tuple[str, str, str]]:
    """
    Query HF ConceptNet backend with caching and normalized output.
    Returns list of (start, rel, end) triples.
    """
    relations = relations or DEFAULT_RELATIONS
    word_norm = _normalize_word(word)
    cache = _load_cache()
    cache_key = _make_cache_key(lang, word_norm, relations)

    if cache_only is None:
        cache_only = os.getenv("CONCEPTNET_CACHE_ONLY", "0") == "1"

    force_refresh = os.getenv("CONCEPTNET_FORCE_REFRESH", "0") == "1"
    if cache_key in cache and not force_refresh:
        print(f"[ConceptNet] cache hit: {cache_key}")
        return _normalize_triples(cache[cache_key])

    if cache_only:
        print(f"[ConceptNet] cache miss (cache-only): {cache_key}")
        return []

    client = _get_client()
    try:
        result = client.predict(
            word=word_norm,
            lang=lang,
            selected_relations=relations,
            api_name="/get_semantic_profile",
        )
    except Exception as e:
        print(f"[ConceptNet] HF query error for '{word_norm}': {e}")
        return []

    triples = _parse_hf_result(word_norm, result, relations)
    cache[cache_key] = triples
    _save_cache(cache)
    print(f"[ConceptNet] fetched: {cache_key} (triples={len(triples)})")
    return _normalize_triples(triples)
