"""Cross-encoder reranking bằng BAAI/bge-reranker-v2-m3.

Khác biệt với bi-encoder (bge-m3 dùng ở Task 4/5):

  Bi-encoder   mã hoá query và document THÀNH HAI vector riêng rồi đo cosine.
               Nhanh, đánh chỉ mục trước được, nhưng model không bao giờ nhìn
               thấy query và document cùng lúc → bỏ lỡ quan hệ tinh tế.

  Cross-encoder đưa CẶP (query, document) vào cùng một lượt forward, cho phép
               attention chạy chéo giữa hai bên. Chính xác hơn hẳn, nhưng phải
               chạy lại cho từng cặp → không thể đánh chỉ mục trước.

Vì vậy quy trình chuẩn là hai tầng: bi-encoder lọc nhanh lấy ~20 ứng viên,
cross-encoder chấm lại kỹ để chọn 5. Chấm lại 20 cặp mất ~1-2 giây trên CPU —
chấp nhận được; chấm cả 302 chunk thì không.

Model được nạp lười (lazy) và cache lại, nên import module này không tốn gì.
"""

from functools import lru_cache

from .config import AdvancedConfig


class RerankerUnavailable(RuntimeError):
    """Không nạp được model rerank (thiếu thư viện, thiếu mạng, hết RAM...)."""


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Nạp model một lần duy nhất cho mỗi tên model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankerUnavailable(
            "Thiếu sentence-transformers. Chạy: pip install sentence-transformers"
        ) from exc

    try:
        # max_length=512: cắt cặp (query, chunk) cho vừa cửa sổ của model.
        return CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        raise RerankerUnavailable(f"Không nạp được model rerank {model_name}: {exc}") from exc


def is_available(model_name: str) -> bool:
    """Kiểm tra nạp được model hay không — dùng cho UI để hiện trạng thái."""
    try:
        _load_cross_encoder(model_name)
        return True
    except RerankerUnavailable:
        return False


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[dict]:
    """Chấm lại độ liên quan của từng ứng viên với query.

    Giữ nguyên điểm fusion cũ ở ``pre_rerank_score`` và thứ hạng cũ ở
    ``pre_rerank_rank`` để UI so sánh được trước/sau — đây là bằng chứng
    trực quan nhất cho thấy rerank có tác dụng.
    """
    if not candidates:
        return []
    if not query or not str(query).strip():
        return candidates[:top_k]

    model = _load_cross_encoder(model_name)

    pairs = [[query, str(item.get("content", ""))] for item in candidates]
    scores = model.predict(pairs)

    scored = []
    for old_rank, (item, score) in enumerate(zip(candidates, scores), start=1):
        result = dict(item)
        result["pre_rerank_score"] = float(item.get("score", 0.0))
        result["pre_rerank_rank"] = old_rank
        result["pre_rerank_score_type"] = item.get("score_type", "unknown")
        result["score"] = float(score)
        result["score_type"] = "cross_encoder"
        result["rerank_model"] = model_name
        scored.append(result)

    scored.sort(key=lambda item: item["score"], reverse=True)
    for new_rank, item in enumerate(scored, start=1):
        item["rank_delta"] = item["pre_rerank_rank"] - new_rank
    return scored[:top_k]


def maybe_rerank(
    query: str,
    candidates: list[dict],
    config: AdvancedConfig,
    top_k: int | None = None,
) -> tuple[list[dict], str]:
    """Rerank nếu được bật và nạp được model; không thì trả nguyên trạng.

    Returns:
        (results, status) — ``status`` để UI hiển thị: "reranked" | "disabled" |
        "unavailable: <lý do>"
    """
    limit = top_k or config.top_k
    if not config.rerank_enabled:
        return candidates[:limit], "disabled"
    try:
        return (
            rerank_cross_encoder(query, candidates, top_k=limit, model_name=config.rerank_model),
            "reranked",
        )
    except RerankerUnavailable as exc:
        # Không được để việc thiếu model làm sập cả chatbot giữa buổi demo.
        return candidates[:limit], f"unavailable: {exc}"
