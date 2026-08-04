"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []  # Global override if provided

_CACHED_CORPUS: list[dict] | None = None
_CACHED_BM25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    global _CACHED_CORPUS
    if _CACHED_CORPUS is not None:
        return _CACHED_CORPUS

    corpus = []
    # Sort files deterministically
    for file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative_path = file.relative_to(STANDARDIZED_DIR).as_posix()
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        for idx, sec in enumerate(sections):
            corpus.append(
                {
                    "content": sec,
                    "metadata": {
                        "source": file.name,
                        "relative_path": relative_path,
                        "type": file.parent.name,
                        "chunk_id": f"{relative_path}#section-{idx}",
                    },
                }
            )
    _CACHED_CORPUS = corpus
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    return BM25Okapi([_tokenize(doc["content"]) for doc in corpus])


def _get_bm25_index(corpus: list[dict]):
    global _CACHED_BM25, _CACHED_CORPUS
    if CORPUS:
        # Dynamic corpus passed by caller
        return build_bm25_index(corpus)

    if _CACHED_BM25 is None:
        _CACHED_BM25 = build_bm25_index(corpus)
    return _CACHED_BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'score_type': 'bm25',
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = CORPUS or _load_corpus()
    tokens = _tokenize(query)
    if top_k <= 0 or not corpus or not tokens:
        return []

    bm25 = _get_bm25_index(corpus)
    scores = bm25.get_scores(tokens)
    results = [
        {
            "content": doc["content"],
            "score": float(score),
            "score_type": "bm25",
            "metadata": dict(doc.get("metadata", {})),
        }
        for doc, score in zip(corpus, scores)
    ]
    # Filter candidates with no query term overlap if max_score > 0
    max_score = max((r["score"] for r in results), default=0.0)
    if max_score > 0:
        results = [r for r in results if r["score"] > 0]
    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]



if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
