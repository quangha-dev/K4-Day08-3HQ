"""Cấu hình lớp nâng cao — đọc từ .env, có giá trị mặc định an toàn."""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class AdvancedConfig:
    """Toàn bộ công tắc của pipeline nâng cao."""

    # --- Fusion ---------------------------------------------------------
    # "rrf"   : chỉ dùng thứ hạng, miễn nhiễm với lệch thang điểm
    # "alpha" : alpha*dense_norm + (1-alpha)*sparse_norm, cần chuẩn hoá trước
    fusion_method: str = "rrf"
    alpha: float = 0.6
    rrf_k: int = 60

    # --- Rerank ---------------------------------------------------------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Lấy rộng rồi mới chấm lại: cross-encoder chỉ cải thiện được thứ hạng
    # trong tập ứng viên, nên tập vào phải đủ rộng mới có gì để cải thiện.
    rerank_candidates: int = 20

    # --- Context --------------------------------------------------------
    # Nhồi context sát trần khiến model bỏ sót phần giữa và tốn tiền vô ích.
    context_budget_ratio: float = 0.6
    model_context_window: int = 128_000
    # Trần tuyệt đối: gpt-4o-mini có cửa sổ 128k nhưng 60% của nó (~77k token)
    # là vô nghĩa cho 5 chunk. Đây mới là giới hạn thực tế có tác dụng.
    max_context_tokens: int = 6_000

    # --- Retrieval ------------------------------------------------------
    top_k: int = 5
    # Dưới ngưỡng này thì chuyển sang PageIndex fallback (so với cosine GỐC).
    score_threshold: float = 0.48
    # Chunk có cosine dưới ngưỡng này bị loại khỏi context. Chunk rác lọt vào
    # context không chỉ vô dụng mà còn khiến LLM trích dẫn nhầm.
    min_chunk_score: float = 0.30
    # Không có chunk nào đạt ngưỡng này thì TỪ CHỐI TRẢ LỜI, không gọi LLM.
    # Đây là cổng cuối chống bịa đặt: không bằng chứng thì không phát ngôn.
    min_evidence_score: float = 0.35

    # --- Guardrail ------------------------------------------------------
    guard_enabled: bool = True
    guard_use_llm: bool = True
    # Hỏi lại khi câu hỏi mới chỉ là chủ đề, chưa nêu rõ muốn biết gì.
    clarify_enabled: bool = True

    def effective_context_tokens(self) -> int:
        """Ngân sách token thực tế cho phần context."""
        by_ratio = int(self.model_context_window * self.context_budget_ratio)
        return max(512, min(by_ratio, self.max_context_tokens))


def load_config(**overrides) -> AdvancedConfig:
    """Đọc cấu hình từ biến môi trường, cho phép ghi đè từng field."""
    config = AdvancedConfig(
        fusion_method=os.getenv("ADV_FUSION", "rrf").strip().lower(),
        alpha=_env_float("ADV_ALPHA", 0.6),
        rrf_k=_env_int("ADV_RRF_K", 60),
        rerank_enabled=_env_bool("ADV_RERANK", False),
        rerank_model=os.getenv("ADV_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidates=_env_int("ADV_RERANK_CANDIDATES", 20),
        context_budget_ratio=_env_float("ADV_CONTEXT_BUDGET", 0.6),
        model_context_window=_env_int("ADV_MODEL_WINDOW", 128_000),
        max_context_tokens=_env_int("ADV_MAX_CONTEXT_TOKENS", 6_000),
        top_k=_env_int("ADV_TOP_K", 5),
        score_threshold=_env_float("ADV_SCORE_THRESHOLD", 0.48),
        min_chunk_score=_env_float("ADV_MIN_CHUNK_SCORE", 0.30),
        min_evidence_score=_env_float("ADV_MIN_EVIDENCE_SCORE", 0.35),
        guard_enabled=_env_bool("ADV_GUARD", True),
        guard_use_llm=_env_bool("ADV_GUARD_LLM", True),
        clarify_enabled=_env_bool("ADV_CLARIFY", True),
    )
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    if config.fusion_method not in {"rrf", "alpha"}:
        config.fusion_method = "rrf"
    config.alpha = min(max(config.alpha, 0.0), 1.0)
    return config
