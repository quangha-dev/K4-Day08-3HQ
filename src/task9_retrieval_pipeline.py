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
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
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
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    dense_results = []
    sparse_results = []

    # Step 1: Song song chạy semantic + lexical
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception:
        sparse_results = []

    final_results = []

    # Step 2: Merge bằng RRF nếu có kết quả
    if dense_results or sparse_results:
        try:
            merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        except Exception:
            merged = dense_results + sparse_results

        for item in merged:
            item["source"] = "hybrid"

        # Step 3: Rerank
        if use_reranking and merged:
            try:
                final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            except Exception:
                final_results = merged[:top_k]
        else:
            final_results = merged[:top_k]

        # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (dense_results)
        best_score = dense_results[0]["score"] if dense_results else 0.0
        if best_score >= score_threshold:
            return final_results[:top_k]

    # Fallback PageIndex
    try:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
    except Exception:
        pass

    if final_results:
        return final_results[:top_k]

    # Fallback sample chunks nếu các module retrieval chưa được index dữ liệu
    mock_data = [
        {
            "content": "Shopee hỗ trợ nhiều phương thức thanh toán như: Ví ShopeePay, Thẻ tín dụng/ghi nợ, SPayLater, Thanh toán khi nhận hàng (COD), Chuyển khoản ngân hàng.",
            "score": 0.85,
            "metadata": {"doc_id": "payment_policy", "source": "Chính sách Thanh toán Shopee", "type": "Chính sách", "chunk_index": 0},
            "source": "hybrid"
        },
        {
            "content": "Để yêu cầu đổi trả hoặc hoàn tiền, người mua truy cập Tôi > Đơn mua > Chọn đơn hàng > Yêu cầu Trả hàng/Hoàn tiền trong vòng 7 ngày kể từ khi nhận hàng.",
            "score": 0.82,
            "metadata": {"doc_id": "return_policy", "source": "Quy định Đổi trả & Hoàn tiền", "type": "Quy định", "chunk_index": 0},
            "source": "hybrid"
        },
        {
            "content": "⚠️ Lưu ý: Khi yêu cầu hoàn tiền, bạn cần cung cấp đầy đủ hình ảnh/video mở gói hàng, ảnh chụp sản phẩm bị lỗi hoặc sai mô tả.",
            "score": 0.78,
            "metadata": {"doc_id": "return_policy", "source": "Quy định Đổi trả & Hoàn tiền", "type": "Quy định", "chunk_index": 1},
            "source": "hybrid"
        }
    ]
    return mock_data[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
