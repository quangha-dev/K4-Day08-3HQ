"""Task 7 — Reciprocal Rank Fusion only."""

import json


def _candidate_identity(item: dict, list_index: int, rank: int) -> str:
    metadata = item.get("metadata") or {}
    if metadata.get("chunk_id"):
        return f"chunk:{metadata['chunk_id']}"
    content = str(item.get("content") or "").strip()
    if content:
        return f"content:{content}"
    stable_metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
    if stable_metadata:
        return f"metadata:{json.dumps(stable_metadata, sort_keys=True, default=str)}"
    return f"position:{list_index}:{rank}"


def _source_name(item: dict, list_index: int, existing: dict) -> str:
    base = item.get("retrieval_source") or item.get("source") or f"ranker_{list_index}"
    name = str(base)
    suffix = 2
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse independently-ranked candidate lists without mixing score scales."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    candidates: dict[str, dict] = {}
    raw_scores: dict[str, dict] = {}
    for list_index, ranked_list in enumerate(ranked_lists):
        seen: set[str] = set()
        unique_rank = 0
        for item_rank, item in enumerate(ranked_list, start=1):
            key = _candidate_identity(item, list_index, item_rank)
            if key in seen:
                continue
            seen.add(key)
            unique_rank += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + unique_rank)
            candidates.setdefault(key, dict(item))
            provenance = raw_scores.setdefault(key, {})
            provenance[_source_name(item, list_index, provenance)] = {
                "score": item.get("score", 0.0),
                "score_type": item.get("score_type", f"ranker_{list_index}"),
            }

    results = []
    for key, score in sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]:
        result = dict(candidates[key])
        result.update(
            score=score,
            score_type="rrf",
            retrieval_source="hybrid",
            source="hybrid",
            raw_scores=raw_scores[key],
        )
        results.append(result)
    return results


def rerank(query: str, candidates: list[dict] | list[list[dict]], top_k: int = 5, method: str = "rrf") -> list[dict]:
    """Run RRF for multiple lists or preserve score ordering for one list."""
    if method != "rrf":
        raise ValueError("Only RRF reranking is implemented; use method='rrf'.")
    if candidates and isinstance(candidates[0], list):
        return rerank_rrf(candidates, top_k)
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
