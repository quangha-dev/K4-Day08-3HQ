"""Pipeline nâng cao — bọc ngoài Task 5/6/8/10, có trace từng tầng.

Luồng:
    query
      ├─ semantic_search  (Task 5, bge-m3, cosine gốc)
      ├─ lexical_search   (Task 6, BM25)
      ├─ fuse             (RRF hoặc Alpha weighting)
      ├─ cross-encoder    (bge-reranker-v2-m3, tuỳ chọn)
      ├─ token budget + reorder + XML + đánh số
      └─ LLM              (gpt-4o-mini, citation theo số)

Mọi tầng đều ghi lại vào ``trace`` để Pipeline Inspector hiển thị được thứ
hạng thay đổi ra sao sau mỗi bước — đây là bằng chứng trực quan cho demo.

QUAN TRỌNG: module này KHÔNG được import bởi tests/test_individual.py và
KHÔNG sửa gì trong src/task*.py, nên 35 test chấm điểm không bị ảnh hưởng.
"""

import os
import time

from ..task5_semantic_search import semantic_search
from ..task6_lexical_search import lexical_search
from ..task8_pageindex_vectorless import pageindex_search
from .config import AdvancedConfig, load_config
from .clarify import ClarifyResult, clarification_message
from .clarify import analyze as clarify_analyze
from .context_builder import build_prompt
from .fusion import fuse
from .guardrails import GuardResult, Verdict, analyze, refusal_message
from .reranker import maybe_rerank

SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu chính sách thương mại điện tử. Bạn chỉ trả lời dựa "
    "trên tài liệu được cung cấp và luôn dẫn nguồn theo số tài liệu. Bạn không "
    "bao giờ tự suy diễn điều kiện không được nêu rõ trong tài liệu."
)

NO_EVIDENCE_ANSWER = (
    "I cannot verify this information — không tìm thấy tài liệu chính sách nào "
    "đủ căn cứ để trả lời câu hỏi này."
)


def _snapshot(results: list[dict], limit: int = 10) -> list[dict]:
    """Ảnh chụp gọn một tầng để hiển thị, không kéo theo cả nội dung dài."""
    rows = []
    for rank, item in enumerate(results[:limit], start=1):
        metadata = item.get("metadata", {}) or {}
        rows.append({
            "rank": rank,
            "score": round(float(item.get("score", 0.0)), 5),
            "score_type": item.get("score_type", ""),
            "source": metadata.get("source", "?"),
            "section": metadata.get("section") or metadata.get("subsection") or "",
            "customer_role": metadata.get("customer_role", "-"),
            "preview": " ".join(str(item.get("content", "")).split())[:110],
            "pre_rerank_rank": item.get("pre_rerank_rank"),
            "rank_delta": item.get("rank_delta"),
            "fusion_detail": item.get("fusion_detail"),
        })
    return rows


def _noop(*_args, **_kwargs) -> None:
    """Callback rỗng — dùng khi không ai theo dõi tiến trình."""


def retrieve_advanced(
    query: str,
    config: AdvancedConfig | None = None,
    on_stage=None,
) -> dict:
    """Chạy toàn bộ tầng truy hồi, trả kết quả kèm trace từng bước.

    Args:
        on_stage: hàm ``(ten_tang, thong_tin) -> None`` được gọi ngay khi mỗi
            tầng bắt đầu và kết thúc. Nhờ đó UI vẽ được tiến trình THẬT theo
            thời gian thực, không phải phát lại hoạt ảnh giả sau khi xong.
    """
    config = config or load_config()
    emit = on_stage or _noop
    trace: dict = {"config": {
        "fusion": config.fusion_method,
        "alpha": config.alpha,
        "rerank": config.rerank_enabled,
        "rerank_model": config.rerank_model if config.rerank_enabled else None,
        "top_k": config.top_k,
        "threshold": config.score_threshold,
    }}

    # --- Tầng 1: dense ---------------------------------------------------
    emit("dense", {"status": "running"})
    started = time.perf_counter()
    try:
        dense = semantic_search(query, top_k=config.rerank_candidates) or []
    except Exception as exc:
        dense, trace["dense_error"] = [], str(exc)
    trace["dense"] = _snapshot(dense)
    trace["dense_ms"] = round((time.perf_counter() - started) * 1000)
    dense_top = float(dense[0]["score"]) if dense else 0.0
    emit("dense", {
        "status": "done", "out": len(dense), "ms": trace["dense_ms"],
        "note": f"cosine cao nhất {dense_top:.3f}",
    })

    # --- Tầng 2: sparse --------------------------------------------------
    emit("sparse", {"status": "running"})
    started = time.perf_counter()
    try:
        sparse = lexical_search(query, top_k=config.rerank_candidates) or []
    except Exception as exc:
        sparse, trace["sparse_error"] = [], str(exc)
    trace["sparse"] = _snapshot(sparse)
    trace["sparse_ms"] = round((time.perf_counter() - started) * 1000)
    emit("sparse", {
        "status": "done", "out": len(sparse), "ms": trace["sparse_ms"],
        "note": f"BM25 cao nhất {float(sparse[0]['score']):.2f}" if sparse else "không khớp từ khoá",
    })

    trace["dense_top_score"] = round(dense_top, 4)

    # --- Lọc chunk yếu ---------------------------------------------------
    # Chunk cosine thấp không chỉ vô dụng mà còn có hại: nó chiếm chỗ trong
    # context và tạo cơ hội cho LLM trích dẫn nhầm nguồn.
    before = len(dense)
    dense = [item for item in dense if float(item.get("score", 0.0)) >= config.min_chunk_score]
    trace["weak_chunks_dropped"] = before - len(dense)
    trace["min_chunk_score"] = config.min_chunk_score

    # --- Tầng 3: fusion --------------------------------------------------
    method_label = "RRF k=60" if config.fusion_method == "rrf" else f"Alpha α={config.alpha}"
    emit("fusion", {"status": "running", "note": method_label})
    started = time.perf_counter()
    fused = fuse(dense, sparse, config, top_k=config.rerank_candidates) if (dense or sparse) else []
    trace["fused"] = _snapshot(fused)
    trace["fusion_ms"] = round((time.perf_counter() - started) * 1000)
    both = sum(1 for item in fused if len(item.get("fusion_detail") or {}) > 1)
    emit("fusion", {
        "status": "done", "in": len(dense) + len(sparse), "out": len(fused),
        "ms": trace["fusion_ms"], "note": f"{method_label} · {both} chunk trúng cả 2 ranker",
    })

    # --- Tầng 4: cross-encoder rerank ------------------------------------
    emit("rerank", {"status": "running" if config.rerank_enabled else "skipped"})
    started = time.perf_counter()
    reranked, rerank_status = maybe_rerank(query, fused, config, top_k=config.top_k)
    trace["reranked"] = _snapshot(reranked)
    trace["rerank_status"] = rerank_status
    trace["rerank_ms"] = round((time.perf_counter() - started) * 1000)
    moved = sum(1 for item in reranked if item.get("rank_delta"))
    emit("rerank", {
        "status": "done" if rerank_status == "reranked" else "skipped",
        "in": len(fused), "out": len(reranked), "ms": trace["rerank_ms"],
        "note": f"{moved}/{len(reranked)} chunk đổi thứ hạng" if rerank_status == "reranked"
                else ("đang tắt" if rerank_status == "disabled" else "model không khả dụng"),
    })

    # --- Tầng 5: fallback khi evidence yếu -------------------------------
    # So với ĐIỂM COSINE GỐC, không bao giờ so với điểm fusion.
    mode = "hybrid"
    results = reranked
    need_fallback = not dense or dense_top < config.score_threshold
    emit("fallback", {"status": "running" if need_fallback else "skipped"})
    if need_fallback:
        try:
            fallback = pageindex_search(query, top_k=config.top_k) or []
        except Exception:
            fallback = []
        if fallback:
            results = fallback
            mode = "pageindex"
            trace["fallback"] = _snapshot(fallback)
    trace["retrieval_mode"] = mode
    trace["fallback_triggered"] = mode == "pageindex"
    emit("fallback", {
        "status": "done" if mode == "pageindex" else "skipped",
        "out": len(results) if mode == "pageindex" else 0,
        "note": f"cosine {dense_top:.3f} < ngưỡng {config.score_threshold}" if need_fallback
                else f"cosine {dense_top:.3f} ≥ ngưỡng {config.score_threshold}, không cần",
    })

    # --- Cổng bằng chứng -------------------------------------------------
    # Không có bằng chứng đủ mạnh thì KHÔNG được trả lời. Đây là ranh giới
    # giữa một hệ thống tra cứu đáng tin và một cỗ máy đoán mò có giọng điệu
    # tự tin. Fallback PageIndex dùng thang điểm khác nên được miễn kiểm tra.
    has_evidence = bool(results)
    if results and mode == "hybrid":
        has_evidence = dense_top >= config.min_evidence_score
    trace["has_evidence"] = has_evidence
    trace["min_evidence_score"] = config.min_evidence_score
    emit("evidence", {
        "status": "done" if has_evidence else "blocked",
        "out": len(results) if has_evidence else 0,
        "note": f"bằng chứng {dense_top:.3f} ≥ {config.min_evidence_score} → cho phép trả lời"
                if has_evidence
                else f"bằng chứng {dense_top:.3f} < {config.min_evidence_score} → TỪ CHỐI trả lời",
    })

    return {"results": results, "trace": trace, "config": config, "has_evidence": has_evidence}


def generate_advanced(
    query: str,
    history: list[dict] | None = None,
    config: AdvancedConfig | None = None,
    model: str | None = None,
    on_stage=None,
) -> dict:
    """Truy hồi nâng cao + sinh câu trả lời có citation theo số."""
    from ..task10_generation import (
        MAX_HISTORY_TURNS,
        TEMPERATURE,
        TOP_P,
        _resolve_llm_provider,
        _normalize_model_id,
    )

    config = config or load_config()
    model = model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    emit = on_stage or _noop

    # === CỔNG 1: Guardrail — chạy TRƯỚC khi tốn bất kỳ tài nguyên nào ======
    # Câu hỏi độc hại không được phép đi vào pipeline. Chặn ở đây nghĩa là
    # không embedding, không truy vấn vector store, không token sinh.
    emit("guard", {"status": "running"})
    guard = analyze(query, history, use_llm=config.guard_use_llm, model=model) \
        if config.guard_enabled else GuardResult(Verdict.ALLOW, "Guardrail đang tắt.")
    emit("guard", {
        "status": "done" if guard.allowed else "blocked",
        "note": f"{guard.verdict.value} · tầng {guard.layer}",
    })

    if not guard.allowed:
        return {
            "answer": refusal_message(guard),
            "sources": [],
            "citation_map": [],
            "trace": {"guard": guard.to_dict(), "blocked_before_rag": True},
            "guard": guard.to_dict(),
            "degraded": False,
            "refused": True,
        }

    # === CỔNG 1.5: Làm rõ câu hỏi =========================================
    # "trả hàng" là chủ đề, không phải câu hỏi. Đem đi embedding thẳng sẽ ra
    # một vector nằm giữa mọi điều khoản, top-5 gom mỗi thứ một ít mà không
    # trúng ý ai. Thông tin còn thiếu nằm ở phía người dùng, nên phải hỏi.
    emit("clarify", {"status": "running"})
    clarify = clarify_analyze(query, history) if config.clarify_enabled else ClarifyResult(False)
    emit("clarify", {
        "status": "blocked" if clarify.needed else "done",
        "note": f"cần làm rõ: {clarify.topic_label}" if clarify.needed else "câu hỏi đã đủ cụ thể",
    })

    if clarify.needed:
        return {
            "answer": clarification_message(clarify),
            "sources": [],
            "citation_map": [],
            "trace": {"guard": guard.to_dict(), "clarify": clarify.to_dict(),
                      "blocked_before_rag": True},
            "guard": guard.to_dict(),
            "clarify": clarify.to_dict(),
            "original_query": query,
            "degraded": False,
            "refused": True,
        }

    # === CỔNG 2: Truy hồi + kiểm tra bằng chứng ===========================
    retrieval = retrieve_advanced(query, config, on_stage=on_stage)
    chunks, trace = retrieval["results"], retrieval["trace"]
    trace["guard"] = guard.to_dict()

    if not chunks or not retrieval.get("has_evidence", False):
        detail = (
            "Không truy hồi được tài liệu nào liên quan."
            if not chunks
            else f"Bằng chứng tốt nhất chỉ đạt {trace.get('dense_top_score', 0):.3f}, "
                 f"dưới ngưỡng tối thiểu {config.min_evidence_score}."
        )
        return {
            "answer": f"{NO_EVIDENCE_ANSWER}\n\n_{detail}_",
            "sources": [],
            "citation_map": [],
            "trace": trace,
            "guard": guard.to_dict(),
            "degraded": False,
            "refused": True,
        }

    emit("context", {"status": "running"})
    prompt = build_prompt(query, chunks, config, model="gpt-4o-mini")
    trace["budget"] = prompt["budget_stats"]
    trace["prompt_tokens"] = prompt["prompt_tokens"]
    budget = prompt["budget_stats"]
    emit("context", {
        "status": "done", "in": len(chunks), "out": budget["kept"],
        "note": f"{budget['used_tokens']}/{budget['budget_tokens']} token"
                + (f" · cắt {budget['dropped']} chunk" if budget["dropped"] else ""),
    })

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        if turn.get("role") in ("user", "assistant") and (turn.get("content") or "").strip():
            messages.append({"role": turn["role"], "content": turn["content"].strip()})
    messages.append({"role": "user", "content": prompt["user_message"]})

    emit("llm", {"status": "running", "note": model})
    llm_started = time.perf_counter()
    answer, degraded = "", False
    try:
        from openai import OpenAI

        api_key, base_url, needs_prefix = _resolve_llm_provider()
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=_normalize_model_id(model, needs_prefix),
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        trace["llm_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    if not answer:
        degraded = True
        lines = ["⚠️ Không gọi được LLM. Trích nguyên văn tài liệu liên quan nhất:", ""]
        for entry in prompt["citation_map"][:3]:
            excerpt = " ".join(str(entry["content"]).split())[:350]
            lines.append(f"[{entry['number']}] {excerpt} — *{entry['source']}*")
            lines.append("")
        answer = "\n".join(lines).strip()

    trace["llm_ms"] = round((time.perf_counter() - llm_started) * 1000)
    emit("llm", {
        "status": "degraded" if degraded else "done",
        "in": len(prompt["citation_map"]), "ms": trace["llm_ms"],
        "note": "chế độ suy giảm — trích nguyên văn" if degraded else f"{model} · {len(answer)} ký tự",
    })

    return {
        "answer": answer,
        "sources": prompt["chunks_used"],
        "citation_map": prompt["citation_map"],
        "trace": trace,
        "guard": guard.to_dict(),
        "degraded": degraded,
        "refused": False,
    }
