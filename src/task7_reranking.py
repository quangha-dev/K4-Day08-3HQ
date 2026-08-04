"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from typing import Optional, Union


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Placeholder).
    """
    raise NotImplementedError("Implement rerank_cross_encoder if cross-encoder method is selected")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse (Placeholder).
    """
    raise NotImplementedError("Call rerank_mmr with query_embedding if MMR method is selected")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    raw_scores_map: dict[str, dict[str, float]] = {}
    candidates: dict[str, dict] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        seen_in_list = set()
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("metadata", {}).get("chunk_id") or item.get("content", "")
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            if key not in raw_scores_map:
                raw_scores_map[key] = {}
            score_type = item.get("score_type", f"ranker_{list_idx}")
            raw_scores_map[key][score_type] = item.get("score", 0.0)

            if key not in candidates:
                candidates[key] = dict(item)

    results = []
    for key, rrf_score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        item = dict(candidates[key])
        item["score"] = rrf_score
        item["score_type"] = "rrf"
        item["raw_scores"] = raw_scores_map[key]
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: Union[list[dict], list[list[dict]]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface. Supports single candidate list or multi-list fusion.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates (hoặc list các danh sách cho RRF)
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
            flat_candidates = [item for sublist in candidates for item in sublist]
            return rerank_cross_encoder(query, flat_candidates, top_k)
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")

    elif method == "rrf":
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
            # True multi-list fusion
            return rerank_rrf(candidates, top_k)
        else:
            # Single list fusion fallback
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            return rerank_rrf([ranked], top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
