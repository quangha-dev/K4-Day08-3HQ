"""Tìm alpha tối ưu cho Alpha Weighting bằng cách quét thử.

Alpha không có giá trị "đúng" phổ quát — nó phụ thuộc corpus và kiểu câu hỏi.
Corpus nhiều thuật ngữ/mã số → sparse quan trọng hơn → alpha thấp. Corpus văn
xuôi, người dùng hỏi bằng từ đồng nghĩa → dense quan trọng hơn → alpha cao.
Cách duy nhất để biết là đo trên chính golden dataset của nhóm.

Chỉ số dùng để đo: Hit Rate và MRR (Mean Reciprocal Rank), tính bằng cách đối
chiếu `expected_context` trong golden_dataset với nguồn/mục của chunk truy hồi.
Đây là chỉ số cho RETRIEVER — cố ý không gọi LLM, nên chạy nhanh và không tốn
tiền API, có thể quét 11 giá trị alpha trong vài chục giây.

Chạy:
    python -m src.advanced.tune_alpha
"""

import json
import re
from pathlib import Path

from ..task5_semantic_search import semantic_search
from ..task6_lexical_search import lexical_search
from .config import load_config
from .fusion import alpha_fuse, rrf_fuse

GOLDEN_PATH = Path(__file__).parent.parent.parent / "group_project" / "evaluation" / "golden_dataset.json"
ALPHA_GRID = [round(i / 10, 1) for i in range(11)]  # 0.0 → 1.0
TOP_K = 5
CANDIDATES = 20

_STOPWORDS = {
    "trang", "muc", "mục", "phần", "phan", "chính", "chinh", "sách", "sach",
    "của", "cua", "và", "va", "the", "and", "for", "policy", "page",
}


def _terms(text: str) -> set[str]:
    tokens = re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _is_hit(chunk: dict, expected_context: str) -> bool:
    """Chunk được coi là trúng nếu chồng lấn đủ nhiều với mô tả context mong đợi."""
    expected = _terms(expected_context)
    if not expected:
        return False
    metadata = chunk.get("metadata", {}) or {}
    haystack = " ".join([
        str(metadata.get("source", "")),
        str(metadata.get("title", "")),
        str(metadata.get("section", "")),
        str(metadata.get("subsection", "")),
        str(chunk.get("content", ""))[:600],
    ])
    overlap = expected & _terms(haystack)
    return len(overlap) / len(expected) >= 0.3


def evaluate(results: list[dict], expected_context: str) -> tuple[int, float]:
    """Trả (hit, reciprocal_rank) cho một câu hỏi."""
    for rank, chunk in enumerate(results, start=1):
        if _is_hit(chunk, expected_context):
            return 1, 1.0 / rank
    return 0, 0.0


def run() -> dict:
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    config = load_config()

    # Truy hồi MỘT LẦN cho mỗi câu hỏi rồi tái dùng cho mọi alpha —
    # embedding là phần đắt nhất, không có lý do gì chạy lại 11 lần.
    cached = []
    print(f"Truy hồi {len(dataset)} câu hỏi (dense + sparse)...")
    for item in dataset:
        question = item["question"]
        try:
            dense = semantic_search(question, top_k=CANDIDATES) or []
        except Exception:
            dense = []
        try:
            sparse = lexical_search(question, top_k=CANDIDATES) or []
        except Exception:
            sparse = []
        cached.append((item.get("expected_context", ""), dense, sparse))

    rows = []
    for alpha in ALPHA_GRID:
        hits = rr_total = 0.0
        for expected, dense, sparse in cached:
            fused = alpha_fuse(dense, sparse, top_k=TOP_K, alpha=alpha)
            hit, rr = evaluate(fused, expected)
            hits += hit
            rr_total += rr
        rows.append({
            "alpha": alpha,
            "hit_rate": round(hits / len(cached), 4),
            "mrr": round(rr_total / len(cached), 4),
        })

    # Mốc so sánh: RRF không cần tuning
    hits = rr_total = 0.0
    for expected, dense, sparse in cached:
        fused = rrf_fuse([dense, sparse], top_k=TOP_K, k=config.rrf_k, labels=["dense", "sparse"])
        hit, rr = evaluate(fused, expected)
        hits += hit
        rr_total += rr
    baseline = {"hit_rate": round(hits / len(cached), 4), "mrr": round(rr_total / len(cached), 4)}

    best = max(rows, key=lambda r: (r["mrr"], r["hit_rate"]))

    print(f"\n{'alpha':>6} | {'hit_rate':>9} | {'mrr':>7}   (alpha=1.0 là dense-only, 0.0 là sparse-only)")
    print("-" * 46)
    for row in rows:
        mark = "  <-- tốt nhất" if row["alpha"] == best["alpha"] else ""
        print(f"{row['alpha']:>6} | {row['hit_rate']:>9} | {row['mrr']:>7}{mark}")
    print("-" * 46)
    print(f"{'RRF':>6} | {baseline['hit_rate']:>9} | {baseline['mrr']:>7}   (không cần tuning)")
    print(f"\nĐề xuất: ADV_FUSION=alpha, ADV_ALPHA={best['alpha']}")
    if baseline["mrr"] >= best["mrr"]:
        print("Lưu ý: RRF đang ngang hoặc tốt hơn alpha tốt nhất → nên giữ RRF cho gọn.")

    return {"grid": rows, "best": best, "rrf_baseline": baseline}


if __name__ == "__main__":
    run()
