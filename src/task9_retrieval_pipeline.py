"""Task 9 — dense/sparse fusion with a dense-cosine-only fallback decision."""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def _normalize_result(item: dict, retrieval_source: str) -> dict:
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("chunk_id", None)
    metadata.setdefault("source_file", metadata.get("source", ""))
    metadata.setdefault("page", None)
    score = float(item.get("score", 0.0))
    score_type = item.get("score_type", "unknown")
    return {
        **item,
        "score": score,
        "score_type": score_type,
        "retrieval_source": retrieval_source,
        "source": retrieval_source,
        "raw_scores": item.get("raw_scores") or {
            retrieval_source: {"score": score, "score_type": score_type}
        },
        "metadata": metadata,
    }


def _prepare_retriever_results(results: list[dict], retrieval_source: str, score_type: str) -> list[dict]:
    prepared = []
    for item in results:
        normalized = _normalize_result(item, retrieval_source)
        normalized["score_type"] = item.get("score_type", score_type)
        prepared.append(normalized)
    return prepared


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Fuse dense and BM25 lists, then fallback using only the original dense cosine."""
    if not isinstance(query, str) or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    try:
        dense_source = semantic_search(query, top_k=top_k * 2)
    except NotImplementedError:
        dense_source = []
    dense_results = _prepare_retriever_results(dense_source, "dense", "cosine")
    sparse_results = _prepare_retriever_results(lexical_search(query, top_k=top_k * 2), "sparse", "bm25")
    best_dense_cosine = max((item["score"] for item in dense_results), default=0.0)

    # Keep rankings separate: RRF must receive two independent ranked lists.
    hybrid_results = rerank(query, [dense_results, sparse_results], top_k=top_k, method="rrf")
    if best_dense_cosine < score_threshold:
        fallback_results = pageindex_search(query, top_k=top_k)
        if fallback_results:
            return [_normalize_result(item, "pageindex") for item in fallback_results]
    return [_normalize_result(item, "hybrid") for item in hybrid_results]
