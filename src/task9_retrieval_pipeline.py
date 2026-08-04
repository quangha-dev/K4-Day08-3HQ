"""Task 9 — dense/sparse fusion with a dense-cosine-only fallback decision."""

import hashlib

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# Ngưỡng so với ĐIỂM COSINE GỐC của semantic_search (thang [0, 1]).
# KHÔNG bao giờ so với điểm RRF đã fuse — điểm RRF luôn ~1/(60+1) ≈ 0.016
# nên fallback sẽ không bao giờ kích hoạt.
SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"
RRF_K = 60


def _fusion_key(content: str) -> str:
    """Khoá hợp nhất dùng chung cho mọi ranker.

    Dense (ChromaDB) và sparse (BM25) dùng hai bộ chunker khác nhau nên
    ``chunk_id`` của hai bên nằm ở hai hệ quy chiếu khác nhau. Nếu để nguyên,
    RRF coi cùng một đoạn văn là hai tài liệu riêng và không bao giờ cộng điểm.
    Chuẩn hoá khoá theo nội dung giúp RRF chỉ hợp nhất khi văn bản thực sự
    trùng nhau, thay vì trùng theo vị trí (dễ gộp nhầm hai đoạn khác nhau).
    """
    normalized = " ".join(str(content or "").split()).lower()
    if not normalized:
        return ""
    return "text:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()


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
        metadata = normalized["metadata"]
        fusion_key = _fusion_key(normalized.get("content", ""))
        if fusion_key:
            # Giữ lại id gốc để trích dẫn, thay id dùng cho fusion.
            metadata["origin_chunk_id"] = metadata.get("chunk_id")
            metadata["chunk_id"] = fusion_key
        prepared.append(normalized)
    return prepared


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF k=60) → Rerank → final_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'score_type': str,
            'metadata': dict,
            'retrieval_source': str,  # 'hybrid' hoặc 'pageindex'
            'source': str,            # giữ tương thích ngược
            'raw_scores': dict,
        }
    """
    # Step 1: Chạy semantic + lexical độc lập. Một nhánh hỏng không được
    # kéo sập nhánh còn lại — pipeline phải suy giảm êm, không crash.
    try:
        dense_raw = semantic_search(query, top_k=top_k * 2) or []
    except Exception:
        dense_raw = []

    try:
        sparse_raw = lexical_search(query, top_k=top_k * 2) or []
    except Exception:
        sparse_raw = []

    dense_results = _prepare_retriever_results(dense_raw, "dense", "cosine")
    sparse_results = _prepare_retriever_results(sparse_raw, "sparse", "bm25")

    final_results: list[dict] = []

    # Step 2 + 3: Gộp thứ hạng bằng RRF rồi rerank.
    if dense_results or sparse_results:
        try:
            merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2, k=RRF_K)
        except Exception:
            # RRF hỏng thì vẫn phải giữ đúng schema: 'source' chỉ được là
            # 'hybrid' hoặc 'pageindex', không được lộ 'dense'/'sparse' ra ngoài.
            merged = []
            for item in sorted(
                dense_results + sparse_results,
                key=lambda entry: entry.get("score", 0.0),
                reverse=True,
            )[: top_k * 2]:
                merged.append({**item, "retrieval_source": "hybrid", "source": "hybrid"})

        if use_reranking and merged:
            try:
                final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            except Exception:
                final_results = merged[:top_k]
        else:
            final_results = merged[:top_k]

        # Step 4: Quyết định fallback DỰA TRÊN ĐIỂM COSINE GỐC.
        best_dense_score = dense_results[0]["score"] if dense_results else 0.0
        if dense_results and best_dense_score >= score_threshold:
            return final_results[:top_k]

    # Step 5: Evidence yếu (hoặc không có dense) → Vectorless fallback.
    try:
        fallback_raw = pageindex_search(query, top_k=top_k) or []
    except Exception:
        fallback_raw = []

    if fallback_raw:
        return _prepare_retriever_results(
            fallback_raw, "pageindex", "pageindex_global_bm25"
        )[:top_k]

    # Fallback không dùng được → trả kết quả hybrid tốt nhất đang có.
    # Không bịa dữ liệu mẫu: thà trả rỗng để Task 10 nói "không đủ thông tin"
    # còn hơn đưa nội dung giả cho LLM trích dẫn.
    return final_results[:top_k]


def retrieve_with_diagnostics(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> dict:
    """Bản mở rộng cho UI: trả thêm số liệu để hiển thị và giải thích quyết định."""
    try:
        dense_raw = semantic_search(query, top_k=top_k * 2) or []
    except Exception:
        dense_raw = []

    results = retrieve(query, top_k=top_k, score_threshold=score_threshold)
    dense_top = float(dense_raw[0]["score"]) if dense_raw else 0.0
    mode = results[0].get("retrieval_source", "hybrid") if results else "none"

    return {
        "results": results,
        "retrieval_mode": mode,
        "dense_top_score": dense_top,
        "threshold": score_threshold,
        "dense_count": len(dense_raw),
        "fallback_triggered": mode == "pageindex",
    }


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Thời hạn yêu cầu trả hàng hoàn tiền là bao lâu?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        diag = retrieve_with_diagnostics(q, top_k=3)
        print(
            f"  mode={diag['retrieval_mode']} | dense_top={diag['dense_top_score']:.3f} "
            f"| threshold={diag['threshold']} | fallback={diag['fallback_triggered']}"
        )
        for i, r in enumerate(diag["results"], 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['retrieval_source']}] {r['content'][:80]}...")
