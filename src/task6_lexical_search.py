"""Task 6 — heading-aware BM25 lexical retrieval."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
MAX_CHUNK_CHARS = 1_000
MIN_CHUNK_CHARS = 24
MAX_HEADING_CONTEXT_CHARS = min(200, MAX_CHUNK_CHARS // 4)
MIN_BODY_BUDGET = min(200, MAX_CHUNK_CHARS // 4)
CORPUS: list[dict] | None = None  # Optional application/test override.

_CACHED_CORPUS: list[dict] | None = None
_CACHED_CORPUS_SIGNATURE: tuple | None = None
_CACHED_BM25: BM25Okapi | None = None
_CACHED_INDEX_SIGNATURE: tuple | None = None
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


# Corpus là tiếng Việt 100%, nhưng người dùng (và bộ câu hỏi đánh giá) hay gõ
# tiếng Anh. BM25 khớp từ theo mặt chữ nên "order tracking" không bao giờ chạm
# được "theo dõi đơn hàng" — recall tụt về 0 dù tài liệu có đúng nội dung.
# Bảng ánh xạ này mở rộng truy vấn sang thuật ngữ tiếng Việt tương đương
# (cross-lingual query expansion). Chỉ mở rộng phía QUERY, không đụng corpus,
# nên điểm BM25 của tài liệu vẫn giữ nguyên ý nghĩa.
_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "order": ("đơn", "hàng"),
    "orders": ("đơn", "hàng"),
    "track": ("theo", "dõi"),
    "tracking": ("theo", "dõi", "vận", "chuyển"),
    "guide": ("hướng", "dẫn"),
    "guides": ("hướng", "dẫn"),
    "instruction": ("hướng", "dẫn"),
    "refund": ("hoàn", "tiền"),
    "refunds": ("hoàn", "tiền"),
    "return": ("trả", "hàng"),
    "returns": ("trả", "hàng"),
    "payment": ("thanh", "toán"),
    "payments": ("thanh", "toán"),
    "method": ("phương", "thức"),
    "methods": ("phương", "thức"),
    "seller": ("người", "bán"),
    "sellers": ("người", "bán"),
    "buyer": ("người", "mua"),
    "buyers": ("người", "mua"),
    "listing": ("đăng", "bán"),
    "regulation": ("quy", "định"),
    "regulations": ("quy", "định"),
    "policy": ("chính", "sách"),
    "policies": ("chính", "sách"),
    "privacy": ("bảo", "mật", "riêng", "tư"),
    "security": ("bảo", "mật"),
    "evidence": ("bằng", "chứng"),
    "proof": ("bằng", "chứng"),
    "shipping": ("vận", "chuyển"),
    "delivery": ("giao", "hàng"),
    "fee": ("phí",),
    "fees": ("phí",),
    "cost": ("chi", "phí"),
    "account": ("tài", "khoản"),
    "product": ("sản", "phẩm"),
    "products": ("sản", "phẩm"),
    "voucher": ("mã", "giảm", "giá"),
    "complaint": ("khiếu", "nại"),
    "dispute": ("tranh", "chấp"),
    "deadline": ("thời", "hạn"),
    "condition": ("điều", "kiện"),
    "conditions": ("điều", "kiện"),
    "support": ("hỗ", "trợ"),
    "fraud": ("lừa", "đảo"),
    "scam": ("lừa", "đảo"),
}


def _expand_query_tokens(tokens: list[str]) -> list[str]:
    """Bổ sung biến thể tiếng Việt cho token truy vấn, giữ nguyên thứ tự gốc."""
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        for alias in _QUERY_ALIASES.get(token, ()):
            if alias not in seen:
                seen.add(alias)
                expanded.append(alias)
    return expanded


def _source_signature() -> tuple:
    return tuple(
        (file.relative_to(STANDARDIZED_DIR).as_posix(), file.stat().st_mtime_ns, file.stat().st_size)
        for file in sorted(STANDARDIZED_DIR.rglob("*.md"))
    )


def _corpus_signature(corpus: list[dict]) -> tuple:
    return tuple(
        (str(item.get("content", "")), json.dumps(item.get("metadata", {}), sort_keys=True, default=str))
        for item in corpus
    )


def invalidate_bm25_cache() -> None:
    """Drop cached corpus and index after a caller changes its source data."""
    global _CACHED_CORPUS, _CACHED_CORPUS_SIGNATURE, _CACHED_BM25, _CACHED_INDEX_SIGNATURE
    _CACHED_CORPUS = None
    _CACHED_CORPUS_SIGNATURE = None
    _CACHED_BM25 = None
    _CACHED_INDEX_SIGNATURE = None


def _split_oversized(text: str, limit: int) -> list[str]:
    parts = []
    remaining = text.strip()
    while len(remaining) > limit:
        boundary = max(remaining.rfind("\n", 0, limit + 1), remaining.rfind(" ", 0, limit + 1))
        if boundary <= 0 or boundary < limit // 2:
            boundary = limit
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _split_body(text: str, limit: int) -> list[str]:
    """Prefer paragraph boundaries before the deterministic hard splitter."""
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        combined = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(combined) <= limit:
            current = combined
            continue
        if current:
            parts.append(current)
            current = ""
        parts.extend(_split_oversized(paragraph, limit))
    if current:
        parts.append(current)
    return parts


def _bounded_heading_context(heading_path: str) -> str:
    if len(heading_path) <= MAX_HEADING_CONTEXT_CHARS:
        return heading_path
    marker = " … "
    remaining = MAX_HEADING_CONTEXT_CHARS - len(marker)
    leading = remaining // 2
    return f"{heading_path[:leading].rstrip()}{marker}{heading_path[-(remaining - leading):].lstrip()}"


def _markdown_chunks(text: str, relative_path: str) -> list[dict]:
    """Split Markdown by heading while retaining its active heading hierarchy."""
    sections: list[tuple[list[str], str]] = []
    headings: list[str] = []
    body: list[str] = []

    def flush(include_empty_heading: bool = False) -> None:
        content = "\n".join(body).strip()
        if content or (include_empty_heading and headings):
            sections.append((list(headings), content))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        flush()
        body = []
        level = len(match.group(1))
        headings = headings[: level - 1] + [f"{'#' * level} {match.group(2)}"]
    flush(include_empty_heading=True)

    rendered: list[tuple[str, str, str]] = []
    for context, body_text in sections:
        heading_path = "\n".join(context)
        prefix = _bounded_heading_context(heading_path)
        if not body_text:
            rendered.append((prefix, heading_path, prefix))
            continue
        available = MAX_CHUNK_CHARS - len(prefix) - (2 if prefix else 0)
        available = max(MIN_BODY_BUDGET, available)
        for part in _split_body(body_text, available):
            content = f"{prefix}\n\n{part}".strip() if prefix else part
            rendered.append((content, heading_path, prefix))

    return [
        {
            "content": content,
            "metadata": {
                "chunk_id": f"{relative_path}#chunk-{index}",
                "heading_path": heading_path,
                "heading_context": heading_context,
            },
        }
        for index, (content, heading_path, heading_context) in enumerate(rendered)
        if content.strip()
    ]


def _load_corpus() -> tuple[list[dict], tuple]:
    global _CACHED_CORPUS, _CACHED_CORPUS_SIGNATURE
    signature = _source_signature()
    if _CACHED_CORPUS is not None and _CACHED_CORPUS_SIGNATURE == signature:
        return _CACHED_CORPUS, signature

    corpus: list[dict] = []
    for file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative_path = file.relative_to(STANDARDIZED_DIR).as_posix()
        for chunk in _markdown_chunks(text, relative_path):
            chunk["metadata"].update(
                {
                    "source": file.name,
                    "source_file": file.name,
                    "relative_path": relative_path,
                    "type": file.parent.name,
                    "page": None,
                }
            )
            corpus.append(chunk)
    _CACHED_CORPUS = corpus
    _CACHED_CORPUS_SIGNATURE = signature
    return corpus, signature


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    return BM25Okapi([_tokenize(doc["content"]) for doc in corpus])


def _active_corpus() -> tuple[list[dict], tuple]:
    if CORPUS is not None:
        return CORPUS, ("override", _corpus_signature(CORPUS))
    return _load_corpus()


def _get_bm25_index(corpus: list[dict], signature: tuple) -> BM25Okapi:
    global _CACHED_BM25, _CACHED_INDEX_SIGNATURE
    if _CACHED_BM25 is None or _CACHED_INDEX_SIGNATURE != signature:
        _CACHED_BM25 = build_bm25_index(corpus)
        _CACHED_INDEX_SIGNATURE = signature
    return _CACHED_BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return lexical-overlap matches, ranked by their raw BM25 score."""
    if not isinstance(query, str) or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []
    tokens = _expand_query_tokens(_tokenize(query))
    corpus, signature = _active_corpus()
    if not tokens or not corpus:
        return []

    token_set = set(tokens)
    index = _get_bm25_index(corpus, signature)
    results = []
    for document, score in zip(corpus, index.get_scores(tokens)):
        if not token_set.intersection(_tokenize(document["content"])):
            continue
        results.append(
            {
                "content": document["content"],
                "score": float(score),
                "score_type": "bm25",
                "retrieval_source": "sparse",
                "raw_scores": {"sparse": {"score": float(score), "score_type": "bm25"}},
                "metadata": dict(document.get("metadata", {})),
            }
        )
    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]
