"""Fusion dense + sparse: RRF và Alpha Weighting.

Bài toán gốc: điểm cosine nằm trong [0, 1], điểm BM25 là điểm thô không chặn
trên (0 → 20+). Cộng thẳng hai loại điểm này là sai về mặt đơn vị đo.

Hai cách xử lý, mỗi cách một đánh đổi:

  RRF   — vứt bỏ điểm, chỉ giữ thứ hạng: score = Σ 1/(k + rank).
          Ưu: miễn nhiễm hoàn toàn với lệch thang đo, không cần tuning.
          Nhược: mất thông tin về *mức độ* liên quan. Tài liệu hạng 1 với
          cosine 0.95 và tài liệu hạng 1 với cosine 0.50 được đối xử như nhau.

  Alpha — chuẩn hoá hai bên về [0, 1] rồi trộn: α·dense + (1-α)·sparse.
          Ưu: giữ được khoảng cách điểm, α tinh chỉnh được theo tập dữ liệu.
          Nhược: kết quả chuẩn hoá phụ thuộc tập trả về (thêm/bớt ứng viên là
          điểm đổi), và phải tự tìm α tối ưu (xem tune_alpha.py).
"""

from .config import AdvancedConfig


def _identity(item: dict) -> str:
    """Khoá hợp nhất — trùng với quy ước ở Task 9 (băm theo nội dung)."""
    metadata = item.get("metadata") or {}
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    content = " ".join(str(item.get("content") or "").split()).lower()
    return f"content:{content}" if content else f"id:{id(item)}"


def _normalize(scores: list[float]) -> list[float]:
    """Đưa điểm về [0, 1] bằng max-normalization, KHÔNG dùng min-max.

    Min-max ``(s-lo)/(hi-lo)`` có một khuyết tật chí mạng với hybrid search:
    tài liệu hạng cuối trong mỗi danh sách luôn bị ép về đúng 0, tức là bị coi
    như hoàn toàn không liên quan. Với danh sách ngắn, một tài liệu hạng 2
    (cosine 0.55 — vẫn khá liên quan) đóng góp 0 điểm, nên nó thua cả tài liệu
    chỉ xuất hiện ở một ranker. Trúng ở cả hai ranker lẽ ra phải là điểm cộng.

    Max-normalization ``s/hi`` giữ nguyên tỉ lệ tương đối: 0.55/0.91 = 0.60,
    phản ánh đúng rằng tài liệu đó liên quan bằng 60% tài liệu tốt nhất.
    Điểm âm (BM25 có thể âm) được dịch lên 0 trước khi chia.
    """
    if not scores:
        return []
    floor = min(0.0, min(scores))
    shifted = [s - floor for s in scores]
    hi = max(shifted)
    if hi < 1e-12:
        return [1.0] * len(scores)
    return [s / hi for s in shifted]


def rrf_fuse(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    labels: list[str] | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion — chỉ dựa vào thứ hạng."""
    labels = labels or [f"ranker_{i}" for i in range(len(ranked_lists))]
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    provenance: dict[str, dict] = {}

    for list_index, ranked in enumerate(ranked_lists):
        label = labels[list_index] if list_index < len(labels) else f"ranker_{list_index}"
        for rank, item in enumerate(ranked, start=1):
            key = _identity(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, dict(item))
            provenance.setdefault(key, {})[label] = {
                "rank": rank,
                "score": float(item.get("score", 0.0)),
                "score_type": item.get("score_type", label),
            }

    ordered = sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]
    results = []
    for key, score in ordered:
        result = dict(items[key])
        result.update(
            score=score,
            score_type="rrf",
            retrieval_source="hybrid",
            source="hybrid",
            fusion_detail=provenance[key],
        )
        results.append(result)
    return results


def alpha_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 5,
    alpha: float = 0.6,
    labels: tuple[str, str] = ("dense", "sparse"),
) -> list[dict]:
    """Alpha weighting — chuẩn hoá min-max hai phía rồi trộn tuyến tính.

    alpha = 1.0 → chỉ dense (thuần ngữ nghĩa)
    alpha = 0.0 → chỉ sparse (thuần từ khoá)
    Tài liệu chỉ xuất hiện ở một phía nhận 0 điểm ở phía còn lại — đây là hành
    vi có chủ đích: trúng ở cả hai ranker phải được thưởng.
    """
    dense_label, sparse_label = labels
    dense_norm = _normalize([float(r.get("score", 0.0)) for r in dense_results])
    sparse_norm = _normalize([float(r.get("score", 0.0)) for r in sparse_results])

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    provenance: dict[str, dict] = {}

    for rank, (item, norm) in enumerate(zip(dense_results, dense_norm), start=1):
        key = _identity(item)
        scores[key] = scores.get(key, 0.0) + alpha * norm
        items.setdefault(key, dict(item))
        provenance.setdefault(key, {})[dense_label] = {
            "rank": rank,
            "score": float(item.get("score", 0.0)),
            "normalized": round(norm, 4),
            "score_type": item.get("score_type", "cosine"),
        }

    for rank, (item, norm) in enumerate(zip(sparse_results, sparse_norm), start=1):
        key = _identity(item)
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) * norm
        items.setdefault(key, dict(item))
        provenance.setdefault(key, {})[sparse_label] = {
            "rank": rank,
            "score": float(item.get("score", 0.0)),
            "normalized": round(norm, 4),
            "score_type": item.get("score_type", "bm25"),
        }

    ordered = sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]
    results = []
    for key, score in ordered:
        result = dict(items[key])
        result.update(
            score=round(score, 6),
            score_type="alpha_weighted",
            retrieval_source="hybrid",
            source="hybrid",
            fusion_detail=provenance[key],
        )
        results.append(result)
    return results


def fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    config: AdvancedConfig,
    top_k: int | None = None,
) -> list[dict]:
    """Điểm vào thống nhất — chọn thuật toán theo cấu hình."""
    limit = top_k or config.top_k
    if config.fusion_method == "alpha":
        return alpha_fuse(dense_results, sparse_results, top_k=limit, alpha=config.alpha)
    return rrf_fuse(
        [dense_results, sparse_results],
        top_k=limit,
        k=config.rrf_k,
        labels=["dense", "sparse"],
    )
