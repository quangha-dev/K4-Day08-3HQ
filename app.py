"""
RAG Chatbot — E-commerce Support
Streamlit app nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption(
        "Trợ lý hỏi đáp chính sách thương mại điện tử "
        "(đổi trả, thanh toán, bảo mật, quy định người bán)"
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Ai chịu chi phí vận chuyển khi hoàn trả sản phẩm?",
        "Điều kiện để được Trả hàng COM là gì?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    use_memory = st.toggle("Nhớ ngữ cảnh hội thoại", value=True)

    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc:** Semantic + BM25 → RRF (k=60) → PageIndex fallback → LLM có citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def render_diagnostics(msg: dict) -> None:
    """Hiển thị pipeline đã quyết định thế nào — phần chứng minh cho demo."""
    mode = msg.get("retrieval_mode")
    if not mode or mode == "none":
        return

    dense_score = msg.get("dense_top_score", 0.0) or 0.0
    threshold = msg.get("threshold", 0.48) or 0.48

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if mode == "pageindex":
            st.warning("🔄 PAGEINDEX FALLBACK", icon="🔄")
        else:
            st.success("⚡ HYBRID (Semantic + BM25 + RRF)", icon="⚡")
    with col_b:
        st.metric("Cosine cao nhất", f"{dense_score:.3f}", f"ngưỡng {threshold}")
    with col_c:
        st.metric("Số nguồn dùng", len(msg.get("sources", [])))

    # Thanh trực quan: evidence mạnh hay yếu so với ngưỡng fallback
    st.progress(min(max(dense_score, 0.0), 1.0))


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            role = meta.get("customer_role", "-")
            section = meta.get("section") or meta.get("subsection") or ""
            score = src.get("score", 0.0)
            score_type = src.get("score_type", "?")

            st.markdown(
                f"**[{i}] {name}** `{doc_type}` · vai trò: `{role}` · "
                f"{score_type}: `{score:.4f}`"
            )
            if section:
                st.caption(f"Mục: {section}")

            raw = src.get("raw_scores") or {}
            if len(raw) > 1:
                detail = " · ".join(
                    f"{k}={v.get('score', 0):.3f}" for k, v in raw.items()
                )
                st.caption(f"🔗 Trúng ở cả 2 ranker → {detail}")

            url = meta.get("source_url")
            if url:
                st.caption(f"[Xem tài liệu gốc]({url})")

            st.text(str(src.get("content", ""))[:300] + "...")
            st.divider()


# =============================================================================
# MAIN
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hỏi đáp chính sách e-commerce — mọi câu trả lời đều kèm nguồn kiểm chứng")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_diagnostics(msg)
            render_sources(msg.get("sources", []))

user_input = st.chat_input("Nhập câu hỏi về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Lịch sử TRƯỚC khi thêm câu hỏi mới — tránh lặp lại chính câu đang hỏi.
    history = (
        [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        if use_memory
        else []
    )

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tổng hợp câu trả lời..."):
            payload = {"retrieval_mode": None, "sources": []}
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k, history=history)
                answer = response.get("answer", "Chưa thể trả lời.")
                payload = {
                    "sources": response.get("sources", []),
                    "retrieval_mode": response.get("retrieval_mode"),
                    "dense_top_score": response.get("dense_top_score", 0.0),
                    "threshold": response.get("threshold", 0.48),
                }
            except NotImplementedError:
                answer = (
                    "⚠️ **Task 10 chưa được implement.** "
                    "Hoàn thành `src/task10_generation.py` để nối pipeline vào UI."
                )
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"

        st.markdown(answer)
        message = {"role": "assistant", "content": answer, **payload}
        render_diagnostics(message)
        render_sources(payload["sources"])

    st.session_state.messages.append(message)
