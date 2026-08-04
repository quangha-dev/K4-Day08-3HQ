"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve, retrieve_with_diagnostics

# Câu trả lời chuẩn khi không đủ bằng chứng (yêu cầu bắt buộc của đề bài).
NO_EVIDENCE_ANSWER = (
    "I cannot verify this information — tôi không tìm thấy nội dung nào trong "
    "bộ tài liệu chính sách hiện có để trả lời câu hỏi này."
)

# Số lượt hội thoại gần nhất được đưa lại vào prompt. Giữ nhỏ để tránh
# đẩy context dài làm loãng phần tài liệu vừa retrieve được.
MAX_HISTORY_TURNS = 6


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM model (OpenRouter model ID). Đổi qua .env mà không phải sửa code.
# Mặc định dùng model ":free" để không phụ thuộc credit — `openai/gpt-4o-mini`
# cần tài khoản có số dư, hết credit sẽ trả 401/402 và làm gãy cả pipeline.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Model id ghi theo kiểu OpenRouter; hàm _normalize_model_id() sẽ tự bỏ tiền tố
# "openai/" khi gọi thẳng API OpenAI.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Thử lần lượt khi model chính không dùng được (hết quota, model bị gỡ...).
LLM_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "LLM_FALLBACK_MODELS",
        "tencent/hy3:free,nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ).split(",")
    if model.strip()
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán và các vấn đề liên quan khác có trong context).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Nếu context không đủ thông tin, trả lời: "Tôi cần thêm thông tin để trả lời câu hỏi này" và yêu cầu cung cấp thêm thông tin
3. Không suy diễn hoặc tự bổ sung chính sách.
4. Khi có nhiều phiên bản chính sách, ưu tiên tài liệu mới nhất.
5. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Returns Policy, 2026]
6. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
7. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect, đồng thời giữ liền mạch
    ngữ cảnh đối với các tài liệu có cấu trúc/nhiều mục (như quy định Shopee).

    Chiến lược (Group-based Lost in the Middle):
    1. Gom các chunks thuộc cùng 1 tài liệu/mục (`doc_id` hoặc `source`) lại với nhau.
    2. Giữ nguyên thứ tự xuất hiện ban đầu (vị trí đoạn văn) trong cùng 1 tài liệu.
    3. Đánh giá điểm ưu tiên của từng nhóm bằng score cao nhất (max score).
    4. Sắp xếp danh sách CÁC NHÓM TÀI LIỆU theo kiểu Lost-in-the-Middle:
       Nhóm quan trọng nhất ở ĐẦU, nhóm quan trọng thứ 2 ở CUỐI, nhóm ít hơn ở GIỮA.

    Args:
        chunks: List of dict sorted by score descending (từ retrieval/reranker)

    Returns:
        List of chunks được reorder để vừa giữ ngữ cảnh mục, vừa tối ưu LLM attention.
    """
    if not chunks or len(chunks) <= 2:
        return chunks

    # Step 1: Gom nhóm chunks theo document ID hoặc source
    groups_map: dict[str, list[dict]] = {}
    group_order: list[str] = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        # Lấy key nhận diện tài liệu/mục
        doc_key = metadata.get("doc_id") or metadata.get("source") or metadata.get("file_name") or "default"
        
        if doc_key not in groups_map:
            groups_map[doc_key] = []
            group_order.append(doc_key)
        groups_map[doc_key].append(chunk)

    # Nếu chỉ có 1 nhóm (tất cả chunks thuộc cùng 1 doc), giữ nguyên thứ tự tự nhiên của doc
    if len(groups_map) == 1:
        # Sắp xếp lại các chunks theo position / chunk_index nếu có metadata
        first_key = group_order[0]
        doc_chunks = groups_map[first_key]
        if any("chunk_index" in c.get("metadata", {}) for c in doc_chunks):
            doc_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        return doc_chunks

    # Step 2: Tính max_score cho mỗi nhóm và sắp xếp danh sách các nhóm giảm dần theo score
    group_scores = []
    for key in group_order:
        max_s = max(c.get("score", 0.0) for c in groups_map[key])
        group_scores.append((key, max_s))
    
    group_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_group_keys = [k for k, _ in group_scores]

    # Step 3: Áp dụng Lost-in-the-Middle cho DANH SÁCH NHÓM
    # Input:  [Group1, Group2, Group3, Group4, Group5] (theo score giảm dần)
    # Output: [Group1, Group3, Group5, Group4, Group2]
    front_groups = sorted_group_keys[::2]       # [1, 3, 5]
    back_groups = sorted_group_keys[1::2][::-1] # [4, 2]
    reordered_group_keys = front_groups + back_groups

    # Step 4: Gộp các chunks từ các nhóm đã reorder
    result_chunks = []
    for key in reordered_group_keys:
        group_chunks = groups_map[key]
        # Trong cùng 1 nhóm, nếu có thông tin chunk_index thì sắp xếp lại theo vị trí đoạn văn gốc
        if any("chunk_index" in c.get("metadata", {}) for c in group_chunks):
            group_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        result_chunks.extend(group_chunks)

    return result_chunks


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể trích dẫn (cite).

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "Không có thông tin context được cung cấp."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        
        source = (
            metadata.get("source")
            or metadata.get("doc_id")
            or metadata.get("file_name")
            or metadata.get("title")
            or f"Document {i}"
        )
        doc_type = metadata.get("type") or metadata.get("category") or "Chính sách"
        year = metadata.get("year") or metadata.get("date") or ""
        year_str = f" | Năm: {year}" if year else ""
        
        header = f"[Tài liệu {i} | Nguồn: {source} | Loại: {doc_type}{year_str}]"
        content = chunk.get("content", "").strip()
        
        context_parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _resolve_llm_provider() -> tuple[str, str, bool]:
    """Chọn endpoint khớp với LOẠI key đang có.

    Key OpenRouter bắt đầu bằng ``sk-or-``; key OpenAI bắt đầu bằng ``sk-``
    (hoặc ``sk-proj-``). Gửi key OpenAI tới endpoint OpenRouter sẽ bị trả 401 —
    lỗi này rất hay gặp vì cả hai đều dùng chung OpenAI SDK nên trông giống nhau.

    Returns:
        (api_key, base_url, needs_vendor_prefix)
        ``needs_vendor_prefix`` = True khi model id phải có dạng ``openai/gpt-4o-mini``
        (OpenRouter), False khi phải là ``gpt-4o-mini`` (OpenAI trực tiếp).
    """
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    # Key nằm nhầm biến môi trường vẫn phải dùng được.
    if openrouter_key.startswith("sk-or-"):
        return openrouter_key, os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL), True
    if openai_key.startswith("sk-or-"):
        return openai_key, os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL), True
    if openai_key:
        return openai_key, os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL), False
    if openrouter_key:
        # Không phải định dạng OpenRouter → nhiều khả năng là key OpenAI đặt nhầm chỗ.
        return openrouter_key, os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL), False

    raise RuntimeError(
        "Chưa có API key. Điền OPENAI_API_KEY (sk-proj-...) hoặc "
        "OPENROUTER_API_KEY (sk-or-v1-...) vào file .env"
    )


def _normalize_model_id(model: str, needs_vendor_prefix: bool) -> str:
    """Chuẩn hoá model id theo endpoint đang dùng."""
    if needs_vendor_prefix:
        return model if "/" in model else f"openai/{model}"
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _extractive_answer(chunks: list[dict], error: Exception | None = None) -> str:
    """Câu trả lời chế độ suy giảm khi không gọi được LLM.

    Không sinh chữ mới — chỉ trích nguyên văn các đoạn đã truy hồi kèm nguồn.
    Nhờ vậy vẫn kiểm chứng được và không có nguy cơ bịa đặt, đúng tinh thần
    "chỉ trả lời khi có nguồn để kiểm chứng".
    """
    if not chunks:
        return NO_EVIDENCE_ANSWER

    lines = [
        "⚠️ Không gọi được mô hình sinh câu trả lời. "
        "Dưới đây là các đoạn tài liệu liên quan nhất, trích nguyên văn:",
        "",
    ]
    for i, chunk in enumerate(chunks[:3], 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("title") or f"Tài liệu {i}"
        section = metadata.get("section") or metadata.get("subsection") or ""
        excerpt = " ".join(str(chunk.get("content", "")).split())[:400]
        label = f"{source}{' — ' + section if section else ''}"
        lines.append(f"{i}. {excerpt} [{label}]")
        lines.append("")

    if error is not None:
        lines.append(f"_Chi tiết lỗi: {type(error).__name__}: {str(error)[:180]}_")
    return "\n".join(lines).strip()


def _build_history_messages(history: list[dict] | None) -> list[dict]:
    """Chuyển lịch sử chat thành messages cho LLM (bỏ metadata thừa).

    Đây là thứ biến chatbot từ hỏi-đáp một lượt thành hội thoại thật:
    câu "còn hàng đông lạnh thì sao?" chỉ hiểu được khi LLM thấy lượt trước.
    """
    if not history:
        return []
    messages = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve (kèm số liệu chẩn đoán để UI hiển thị)
    diagnostics = retrieve_with_diagnostics(query, top_k=top_k)
    chunks = diagnostics["results"]

    # Không có evidence → không gọi LLM. Gọi LLM với context rỗng chỉ tạo cơ hội
    # cho model bịa ra chính sách không tồn tại.
    if not chunks:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
            "retrieval_mode": "none",
            "dense_top_score": diagnostics["dense_top_score"],
            "threshold": diagnostics["threshold"],
            "fallback_triggered": diagnostics["fallback_triggered"],
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    # Dùng XML tags để LLM phân biệt rõ ràng context vs câu hỏi,
    # tránh LLM nhầm lẫn phần context là câu hỏi hoặc là lời yêu cầu.
    # Nhắc lại yêu cầu citation và giới hạn trả lời trong phần instruction
    # giúp LLM "nhớ" rules quan trọng ngay trước khi sinh câu trả lời.
    user_message = f"""\
    Dưới đây là các tài liệu chính sách liên quan được tìm kiếm từ cơ sở dữ liệu:

    <context>
        {context}
    </context>

    Câu hỏi của người dùng:

    <question>
        {query}
    </question>

    Yêu cầu khi trả lời:
    - Chỉ sử dụng thông tin có trong <context> ở trên, không bịa đặt thêm.
    - Mỗi khẳng định phải kèm trích dẫn nguồn ngay sau, ví dụ: [Chính sách Thanh toán Shopee]
    - Nếu context không đủ thông tin, trả lời: "Tôi chưa tìm thấy thông tin về vấn đề này trong tài liệu hiện có."
    - Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc rõ ràng.\
    """


    # Step 5: Call LLM (OpenRouter / OpenAI API)
    from openai import OpenAI

    api_key, base_url, model_prefix = _resolve_llm_provider()
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_build_history_messages(history))
    messages.append({"role": "user", "content": user_message})

    answer = ""
    degraded = False
    last_error: Exception | None = None

    candidate_models = [LLM_MODEL] if not model_prefix else [LLM_MODEL, *LLM_FALLBACK_MODELS]
    for raw_model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=_normalize_model_id(raw_model, model_prefix),
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                break
        except Exception as exc:  # hết quota, 401/402/429, model bị gỡ...
            last_error = exc
            continue

    # Không model nào trả lời được → chế độ suy giảm: trình bày thẳng evidence
    # đã truy hồi kèm trích dẫn, thay vì để cả chatbot gãy giữa buổi demo.
    if not answer:
        degraded = True
        answer = _extractive_answer(reordered, last_error)

    # Step 6: Return
    mode = reordered[0].get("retrieval_source") or reordered[0].get("source", "hybrid")
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": mode,
        "retrieval_mode": mode,
        "dense_top_score": diagnostics["dense_top_score"],
        "threshold": diagnostics["threshold"],
        "fallback_triggered": diagnostics["fallback_triggered"],
        "degraded": degraded,
    }


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
