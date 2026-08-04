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

from .task9_retrieval_pipeline import retrieve


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

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit


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

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
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
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

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
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid") if reordered else "none"
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
