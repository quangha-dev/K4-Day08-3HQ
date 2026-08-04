"""Sơ đồ luồng pipeline kiểu n8n — render bằng HTML/SVG nhúng trong Streamlit.

Vì sao không dùng React/Next.js:
    Pipeline chạy trong Python. Tách giao diện sang framework riêng đồng nghĩa
    phải dựng thêm một tầng API chỉ để đẩy trạng thái sang, cộng thêm build
    step và một tiến trình nữa phải chạy lúc demo. Streamlit đã cho phép nhúng
    HTML/CSS/SVG tuỳ ý, mà callback ``on_stage`` thì cập nhật được ngay trong
    tiến trình — đủ để vẽ tiến trình THẬT, không phải hoạt ảnh phát lại.

Trạng thái mỗi khối:
    idle      chưa chạy tới
    running   đang chạy (có hiệu ứng nhấp nháy)
    done      xong
    skipped   bị bỏ qua (ví dụ rerank đang tắt, fallback không cần)
    blocked   chặn tại đây, pipeline dừng
"""

STAGES = [
    {"id": "guard", "icon": "🛡️", "title": "Guardrail", "sub": "phân loại an toàn"},
    {"id": "clarify", "icon": "❓", "title": "Làm rõ", "sub": "đủ cụ thể chưa?"},
    {"id": "dense", "icon": "🧠", "title": "Dense Search", "sub": "bge-m3 · cosine"},
    {"id": "sparse", "icon": "🔤", "title": "Sparse Search", "sub": "BM25 · từ khoá"},
    {"id": "fusion", "icon": "🔀", "title": "Fusion", "sub": "gộp thứ hạng"},
    {"id": "rerank", "icon": "🎯", "title": "Rerank", "sub": "cross-encoder"},
    {"id": "fallback", "icon": "🔄", "title": "Fallback", "sub": "PageIndex"},
    {"id": "evidence", "icon": "⚖️", "title": "Cổng bằng chứng", "sub": "đủ căn cứ?"},
    {"id": "context", "icon": "📦", "title": "Context", "sub": "budget · reorder"},
    {"id": "llm", "icon": "✨", "title": "Sinh câu trả lời", "sub": "gpt-4o-mini"},
]

_COLORS = {
    "idle": ("#2b2f3a", "#4a5163", "#8b93a7"),
    "running": ("#1e3a5f", "#4da3ff", "#bcdcff"),
    "done": ("#173a2b", "#3ecf8e", "#c6f5e0"),
    "skipped": ("#2b2f3a", "#5a6070", "#7f8699"),
    "blocked": ("#4a1d1d", "#ff5d5d", "#ffc9c9"),
    "degraded": ("#463315", "#ffb020", "#ffe0a3"),
}

_CSS = """
<style>
  .flowwrap{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:#0f1116;border-radius:12px;padding:18px 14px;overflow-x:auto}
  .flowrow{display:flex;align-items:center;gap:0;min-width:max-content}
  .node{border-radius:10px;border:1.5px solid;padding:9px 12px;min-width:132px;
    max-width:170px;transition:all .35s ease}
  .node .hd{display:flex;align-items:center;gap:6px;font-size:12.5px;font-weight:650;
    line-height:1.25}
  .node .sub{font-size:10px;opacity:.75;margin-top:2px}
  .node .metric{font-size:10.5px;margin-top:6px;font-variant-numeric:tabular-nums}
  .node .note{font-size:9.5px;margin-top:3px;opacity:.85;line-height:1.35}
  .run{animation:pulse 1s ease-in-out infinite}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(77,163,255,.55)}
    50%{box-shadow:0 0 0 7px rgba(77,163,255,0)}}
  .conn{width:26px;height:2px;background:#3a4152;position:relative;flex:none}
  .conn.on{background:#3ecf8e}
  .conn::after{content:"";position:absolute;right:-1px;top:-3px;border-left:6px solid #3a4152;
    border-top:4px solid transparent;border-bottom:4px solid transparent}
  .conn.on::after{border-left-color:#3ecf8e}
  .branch{display:flex;flex-direction:column;gap:8px}
  .pill{display:inline-block;font-size:9px;padding:1px 6px;border-radius:99px;
    background:rgba(255,255,255,.09);margin-left:4px}
</style>
"""


def _node_html(stage: dict, state: dict) -> str:
    status = state.get("status", "idle")
    bg, border, text = _COLORS.get(status, _COLORS["idle"])
    cls = "node run" if status == "running" else "node"

    badge = {"running": "⏳", "done": "✓", "skipped": "–", "blocked": "✕", "degraded": "!"}.get(status, "")

    metric = ""
    if state.get("out") is not None and status in ("done", "blocked", "degraded"):
        arrow = f"{state['in']} → {state['out']}" if state.get("in") is not None else f"{state['out']} chunk"
        metric = f'<div class="metric">{arrow}'
        if state.get("ms") is not None:
            metric += f'<span class="pill">{state["ms"]} ms</span>'
        metric += "</div>"
    elif state.get("ms") is not None and status == "done":
        metric = f'<div class="metric"><span class="pill">{state["ms"]} ms</span></div>'

    note = f'<div class="note">{state["note"]}</div>' if state.get("note") else ""

    return (
        f'<div class="{cls}" style="background:{bg};border-color:{border};color:{text}">'
        f'<div class="hd"><span>{stage["icon"]}</span><span>{stage["title"]}</span>'
        f'<span style="margin-left:auto;opacity:.9">{badge}</span></div>'
        f'<div class="sub">{stage["sub"]}</div>{metric}{note}</div>'
    )


def _conn(active: bool) -> str:
    return f'<div class="conn{" on" if active else ""}"></div>'


def render_flow(states: dict) -> str:
    """Dựng HTML sơ đồ luồng từ trạng thái hiện tại của từng khối."""
    done = {"done", "skipped", "degraded"}

    def st_of(sid: str) -> dict:
        return states.get(sid, {"status": "idle"})

    parts = [_CSS, '<div class="flowwrap"><div class="flowrow">']

    # Khối đầu vào
    parts.append(
        '<div class="node" style="background:#232838;border-color:#5c6480;color:#cdd4e6">'
        '<div class="hd"><span>💬</span><span>Câu hỏi</span></div>'
        '<div class="sub">đầu vào người dùng</div></div>'
    )
    parts.append(_conn(st_of("guard").get("status") in done | {"running", "blocked"}))

    # Guardrail
    guard = STAGES[0]
    parts.append(_node_html(guard, st_of("guard")))

    if st_of("guard").get("status") == "blocked":
        parts.append(_conn(False))
        parts.append(
            '<div class="node" style="background:#4a1d1d;border-color:#ff5d5d;color:#ffc9c9">'
            '<div class="hd"><span>🚫</span><span>Dừng</span></div>'
            '<div class="sub">pipeline KHÔNG chạy</div>'
            '<div class="note">không embedding, không truy vấn, không token</div></div>'
        )
        parts.append("</div></div>")
        return "".join(parts)

    # Làm rõ câu hỏi — cũng có thể dừng pipeline tại đây
    clarify_state = st_of("clarify")
    parts.append(_conn(clarify_state.get("status") in done | {"running", "blocked"}))
    parts.append(_node_html(STAGES[1], clarify_state))

    if clarify_state.get("status") == "blocked":
        parts.append(_conn(False))
        parts.append(
            '<div class="node" style="background:#463315;border-color:#ffb020;color:#ffe0a3">'
            '<div class="hd"><span>💬</span><span>Hỏi lại</span></div>'
            '<div class="sub">chờ người dùng chọn</div>'
            '<div class="note">chưa truy xuất — thiếu thông tin ở phía người hỏi</div></div>'
        )
        parts.append("</div></div>")
        return "".join(parts)

    parts.append(_conn(st_of("dense").get("status") in done | {"running"}))

    # Nhánh song song dense / sparse
    parts.append('<div class="branch">')
    parts.append(_node_html(STAGES[2], st_of("dense")))
    parts.append(_node_html(STAGES[3], st_of("sparse")))
    parts.append("</div>")

    # Các khối còn lại nối tiếp
    for stage in STAGES[4:]:
        state = st_of(stage["id"])
        parts.append(_conn(state.get("status") in done | {"running", "blocked"}))
        parts.append(_node_html(stage, state))
        if state.get("status") == "blocked":
            parts.append(_conn(False))
            parts.append(
                '<div class="node" style="background:#4a1d1d;border-color:#ff5d5d;color:#ffc9c9">'
                '<div class="hd"><span>🚫</span><span>Từ chối</span></div>'
                '<div class="sub">không đủ bằng chứng</div>'
                '<div class="note">thà không trả lời còn hơn đoán</div></div>'
            )
            parts.append("</div></div>")
            return "".join(parts)

    parts.append(_conn(st_of("llm").get("status") in done))
    parts.append(
        '<div class="node" style="background:#232838;border-color:#5c6480;color:#cdd4e6">'
        '<div class="hd"><span>📝</span><span>Trả lời</span></div>'
        '<div class="sub">kèm citation [1][2]</div></div>'
    )
    parts.append("</div></div>")
    return "".join(parts)


def initial_states() -> dict:
    """Trạng thái ban đầu — mọi khối đều chưa chạy."""
    return {stage["id"]: {"status": "idle"} for stage in STAGES}


# =============================================================================
# HÌNH ẢNH HOÁ ĐẢO THỨ HẠNG SAU RERANK
# =============================================================================

def render_rerank_movement(rows: list[dict]) -> str:
    """Vẽ hai cột trước/sau, nối bằng đường cong theo hướng dịch chuyển.

    Đây là bằng chứng trực quan nhất cho câu hỏi "cross-encoder làm được gì":
    cùng một tập ứng viên, chỉ khác thứ tự, và thấy rõ chunk nào được kéo lên.
    """
    pairs = [(r.get("pre_rerank_rank"), r["rank"], r) for r in rows if r.get("pre_rerank_rank")]
    if not pairs:
        return (
            '<div style="padding:14px;background:#1a1d26;border-radius:10px;color:#8b93a7;'
            'font-family:sans-serif;font-size:13px">Bật <b>Cross-encoder rerank</b> ở thanh bên '
            'để xem thứ hạng thay đổi thế nào.</div>'
        )

    row_h, top, left_x, right_x, width = 46, 34, 30, 330, 560
    height = top + row_h * max(len(pairs), max(p[0] for p in pairs)) + 16

    svg = [f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(
        f'<text x="{left_x}" y="20" fill="#8b93a7" font-size="12" font-family="sans-serif">'
        f'TRƯỚC (sau fusion)</text>'
        f'<text x="{right_x}" y="20" fill="#8b93a7" font-size="12" font-family="sans-serif">'
        f'SAU (cross-encoder)</text>'
    )

    for old_rank, new_rank, row in sorted(pairs, key=lambda p: p[0]):
        y1 = top + (old_rank - 1) * row_h + 14
        y2 = top + (new_rank - 1) * row_h + 14
        delta = old_rank - new_rank
        color = "#3ecf8e" if delta > 0 else ("#ff6b6b" if delta < 0 else "#5a6070")
        mid = (left_x + 150 + right_x) / 2
        svg.append(
            f'<path d="M {left_x+150} {y1} C {mid} {y1}, {mid} {y2}, {right_x-8} {y2}" '
            f'stroke="{color}" stroke-width="2" fill="none" opacity="0.75"/>'
        )
        label = f"▲{delta}" if delta > 0 else (f"▼{abs(delta)}" if delta < 0 else "—")
        preview = (row.get("preview") or "")[:22]
        source = (row.get("source") or "")[:20]

        svg.append(
            f'<rect x="{left_x}" y="{y1-14}" width="150" height="30" rx="6" '
            f'fill="#232838" stroke="#3a4152"/>'
            f'<text x="{left_x+8}" y="{y1-1}" fill="#cdd4e6" font-size="11" '
            f'font-family="sans-serif">#{old_rank} {source}</text>'
            f'<text x="{left_x+8}" y="{y1+11}" fill="#7f8699" font-size="9" '
            f'font-family="sans-serif">{preview}</text>'
        )
        svg.append(
            f'<rect x="{right_x}" y="{y2-14}" width="160" height="30" rx="6" '
            f'fill="#173a2b" stroke="{color}"/>'
            f'<text x="{right_x+8}" y="{y2-1}" fill="#c6f5e0" font-size="11" '
            f'font-family="sans-serif">#{new_rank} {source}</text>'
            f'<text x="{right_x+8}" y="{y2+11}" fill="{color}" font-size="9.5" '
            f'font-family="sans-serif" font-weight="700">{label} · '
            f'{row.get("score", 0):.3f}</text>'
        )

    svg.append("</svg>")
    return (
        '<div style="background:#0f1116;border-radius:12px;padding:12px">'
        + "".join(svg)
        + "</div>"
    )
