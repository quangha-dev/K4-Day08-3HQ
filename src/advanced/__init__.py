"""Lớp RAG nâng cao — TÁCH BIỆT hoàn toàn khỏi Task 1–10.

Vì sao tách riêng:
    `tests/test_individual.py` chấm điểm dựa trên `src/task5..task10`. Mọi thay
    đổi trong các file đó đều có nguy cơ làm rớt test. Toàn bộ kỹ thuật nâng cao
    (alpha weighting, cross-encoder rerank, token budget, numbered citation)
    nằm ở đây và *bọc ngoài* các hàm gốc, không sửa vào chúng.

    Không file test nào import `src.advanced`, và các model nặng chỉ được nạp
    khi thực sự gọi tới (lazy load). Chạy pytest → không tải model → vẫn nhanh.

Bật/tắt bằng .env:
    ADV_FUSION=rrf|alpha        (mặc định rrf — giống Task 9)
    ADV_ALPHA=0.6               (trọng số dense khi dùng alpha weighting)
    ADV_RERANK=0|1              (bật cross-encoder)
    ADV_RERANK_MODEL=BAAI/bge-reranker-v2-m3
    ADV_CONTEXT_BUDGET=0.6      (tỉ lệ token budget tối đa cho context)
"""

from .config import AdvancedConfig, load_config

__all__ = ["AdvancedConfig", "load_config"]
