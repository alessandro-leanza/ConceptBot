import atexit
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai


EMBEDDING_CACHE_PATH = Path("cache/embedding_cache.json")
KEYWORD_CACHE_PATH = Path("cache/keyword_cache.json")
SIMILARITY_CACHE_PATH = Path("cache/similarity_cache.json")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
CACHE_FLUSH_INTERVAL = int(os.getenv("SEMANTIC_CACHE_FLUSH_INTERVAL", "25"))

_IN_MEMORY_CACHES = {}
_DIRTY_FLAGS = {}
_WRITE_COUNTS = {}


def get_openai_client() -> openai.OpenAI:
    return openai.OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _get_cache(path: Path) -> dict:
    key = str(path)
    if key not in _IN_MEMORY_CACHES:
        _IN_MEMORY_CACHES[key] = _load_json(path)
        _DIRTY_FLAGS[key] = False
        _WRITE_COUNTS[key] = 0
    return _IN_MEMORY_CACHES[key]


def _mark_dirty(path: Path) -> None:
    key = str(path)
    _DIRTY_FLAGS[key] = True
    _WRITE_COUNTS[key] = _WRITE_COUNTS.get(key, 0) + 1
    if _WRITE_COUNTS[key] >= CACHE_FLUSH_INTERVAL:
        flush_cache(path)


def flush_cache(path: Path) -> None:
    key = str(path)
    if key in _IN_MEMORY_CACHES and _DIRTY_FLAGS.get(key):
        _save_json(path, _IN_MEMORY_CACHES[key])
        _DIRTY_FLAGS[key] = False
        _WRITE_COUNTS[key] = 0


def flush_all_caches() -> None:
    flush_cache(EMBEDDING_CACHE_PATH)
    flush_cache(KEYWORD_CACHE_PATH)
    flush_cache(SIMILARITY_CACHE_PATH)


atexit.register(flush_all_caches)


def log_openai_call(kind: str, text: str, duration: float) -> None:
    if os.getenv("SEMANTIC_CACHE_VERBOSE", "0") == "1":
        preview = text.replace("\n", " ")[:80]
        print(f"[semantic-cache] {kind} duration={duration:.2f}s text={preview}")


def _text_key(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _embedding_key(model: str, text: str) -> str:
    return _text_key(model, text.strip())


def get_cached_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    cache = _get_cache(EMBEDDING_CACHE_PATH)
    key = _embedding_key(model, text)
    if key in cache:
        return np.array(cache[key], dtype=float)
    client = get_openai_client()
    start = time.monotonic()
    response = client.embeddings.create(input=text, model=model)
    log_openai_call("embedding", text, time.monotonic() - start)
    embedding = response.data[0].embedding
    cache[key] = embedding
    _mark_dirty(EMBEDDING_CACHE_PATH)
    return np.array(embedding, dtype=float)


def get_cached_keywords(
    text: str,
    model: str = "gpt-4o-mini",
    llm_temperature: float = 0,
) -> List[str]:
    cache = _get_cache(KEYWORD_CACHE_PATH)
    key = _text_key(model, text.strip())
    if key in cache:
        return cache[key]
    prompt = f"Extract the most important keywords (max 2) from the following text (must be single words):\n\n{text}"
    client = get_openai_client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Pay attention to the meaning of the sentence."},
            {"role": "user", "content": prompt},
        ],
        temperature=llm_temperature,
    )
    log_openai_call("keyword", text, time.monotonic() - start)
    keywords = response.choices[0].message.content.strip()
    keywords_list = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
    cache[key] = keywords_list
    _mark_dirty(KEYWORD_CACHE_PATH)
    return keywords_list


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _relations_signature(relations: List[Tuple[str, str, str]]) -> str:
    material = "|".join(["::".join(r) for r in sorted(relations)])
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _similarity_key(kind: str, anchor: str, query: str, relations: List[Tuple[str, str, str]]) -> str:
    return f"{kind}:{_text_key('anchor', anchor)}:{_text_key('query', query)}:{_relations_signature(relations)}"


def get_cached_ope_similarities(
    query: str,
    relations: List[Tuple[str, str, str]],
    targets: List[str],
    kind: str,
    model: str = "text-embedding-ada-002",
) -> List[Tuple[Tuple[str, str, str], float]]:
    cache = _get_cache(SIMILARITY_CACHE_PATH)
    key = _similarity_key(kind, "|".join(sorted(targets)), query, relations)
    if key in cache:
        return [(tuple(item["relation"]), float(item["similarity"])) for item in cache[key]]

    target_embeddings = {target: get_cached_embedding(target, model=model) for target in targets}
    output = []
    for relation in relations:
        relation_text = f"{relation[0]} {relation[1]} {relation[2]}"
        relation_embedding = get_cached_embedding(relation_text, model=model)
        max_similarity = max(cosine_similarity(relation_embedding, target_emb) for target_emb in target_embeddings.values())
        output.append((relation, max_similarity))

    cache[key] = [{"relation": list(relation), "similarity": similarity} for relation, similarity in output]
    _mark_dirty(SIMILARITY_CACHE_PATH)
    return output


def get_cached_urp_request_similarities(
    instruction: str,
    query: str,
    relations: List[Tuple[str, str, str]],
    kind: str,
    model: str = "text-embedding-ada-002",
) -> List[Tuple[Tuple[str, str, str], float]]:
    cache = _get_cache(SIMILARITY_CACHE_PATH)
    key = _similarity_key(kind, instruction, query, relations)
    if key in cache:
        return [(tuple(item["relation"]), float(item["similarity"])) for item in cache[key]]

    request_embedding = get_cached_embedding(instruction, model=model)
    output = []
    for relation in relations:
        relation_text = f"{relation[0]} {relation[1]} {relation[2]}"
        relation_embedding = get_cached_embedding(relation_text, model=model)
        similarity = cosine_similarity(relation_embedding, request_embedding)
        output.append((relation, similarity))

    cache[key] = [{"relation": list(relation), "similarity": similarity} for relation, similarity in output]
    _mark_dirty(SIMILARITY_CACHE_PATH)
    return output


def get_cached_urp_object_keyword_similarities(
    instruction: str,
    query: str,
    relations: List[Tuple[str, str, str]],
    keywords: List[str],
    kind: str,
    model: str = "text-embedding-ada-002",
) -> List[Tuple[Tuple[str, str, str], float]]:
    cache = _get_cache(SIMILARITY_CACHE_PATH)
    key = _similarity_key(kind, instruction, f"{query}|{'|'.join(sorted(keywords))}", relations)
    if key in cache:
        return [(tuple(item["relation"]), float(item["similarity"])) for item in cache[key]]

    keyword_embeddings = [get_cached_embedding(keyword, model=model) for keyword in keywords]
    output = []
    for relation in relations:
        relation_text = f"{relation[0]} {relation[1]} {relation[2]}"
        relation_embedding = get_cached_embedding(relation_text, model=model)
        max_similarity = max(cosine_similarity(relation_embedding, keyword_emb) for keyword_emb in keyword_embeddings)
        output.append((relation, max_similarity))

    cache[key] = [{"relation": list(relation), "similarity": similarity} for relation, similarity in output]
    _mark_dirty(SIMILARITY_CACHE_PATH)
    return output


def precompute_all_similarity_caches() -> Dict[str, str]:
    return {
        "embedding_cache": str(EMBEDDING_CACHE_PATH),
        "keyword_cache": str(KEYWORD_CACHE_PATH),
        "similarity_cache": str(SIMILARITY_CACHE_PATH),
    }
