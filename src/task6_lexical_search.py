"""Task 6 — heading-aware BM25 lexical retrieval."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
MAX_CHUNK_CHARS = 1_000
MIN_CHUNK_CHARS = 24
CORPUS: list[dict] | None = None  # Optional application/test override.

_CACHED_CORPUS: list[dict] | None = None
_CACHED_CORPUS_SIGNATURE: tuple | None = None
_CACHED_BM25: BM25Okapi | None = None
_CACHED_INDEX_SIGNATURE: tuple | None = None
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


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
        if boundary <= 0:
            boundary = limit
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _markdown_chunks(text: str, relative_path: str) -> list[dict]:
    """Split Markdown by heading while retaining its active heading hierarchy."""
    sections: list[tuple[list[str], str]] = []
    headings: list[str] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
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
    flush()

    rendered: list[str] = []
    for context, body_text in sections:
        prefix = "\n".join(context)
        available = max(1, MAX_CHUNK_CHARS - len(prefix) - (2 if prefix else 0))
        for part in _split_oversized(body_text, available):
            rendered.append(f"{prefix}\n\n{part}".strip() if prefix else part)

    chunks: list[str] = []
    pending_small = ""
    for content in rendered:
        if len(content) < MIN_CHUNK_CHARS:
            if chunks:
                chunks[-1] = f"{chunks[-1]}\n\n{content}"
            else:
                pending_small = f"{pending_small}\n\n{content}".strip()
        elif pending_small:
            chunks.append(f"{pending_small}\n\n{content}")
            pending_small = ""
        else:
            chunks.append(content)
    if pending_small:
        chunks.append(pending_small)

    return [
        {
            "content": content,
            "metadata": {"chunk_id": f"{relative_path}#chunk-{index}"},
        }
        for index, content in enumerate(chunks)
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
    tokens = _tokenize(query)
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
