"""Task 8 — bounded PageIndex fallback with local global reranking."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_DOCUMENT_IDS = tuple(
    value.strip() for value in os.getenv("PAGEINDEX_DOCUMENT_IDS", "").split(",") if value.strip()
)
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
DOCUMENT_IDS_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
POLL_INTERVAL_SECONDS = 2
PER_DOCUMENT_TIMEOUT_SECONDS = 60
MAX_CANDIDATES_PER_DOCUMENT = 10
MAX_GLOBAL_CANDIDATES = 40


def _cached_document_ids() -> dict[str, str]:
    if not DOCUMENT_IDS_PATH.exists():
        return {}
    try:
        return json.loads(DOCUMENT_IDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _configured_document_ids() -> tuple[str, ...]:
    return PAGEINDEX_DOCUMENT_IDS or tuple(_cached_document_ids().values())


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _normalize_completed_results(retrieval: dict, document_id: str, source_file: str) -> list[dict]:
    """Normalize only a confirmed successful PageIndex retrieval response."""
    results = []
    for node in retrieval.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []):
            for item in group:
                content = str(item.get("relevant_content", "")).strip()
                if not content or len(results) >= MAX_CANDIDATES_PER_DOCUMENT:
                    continue
                rank = len(results) + 1
                page = item.get("page") or item.get("page_number") or item.get("page_label")
                results.append(
                    {
                        "content": content,
                        "score": float(rank),  # Local rank only; never used as global relevance.
                        "score_type": "pageindex_rank",
                        "retrieval_source": "pageindex",
                        "source": "pageindex",
                        "metadata": {
                            "document_id": document_id,
                            "source_file": source_file,
                            "page": page,
                            "section": item.get("section_title") or item.get("title") or "",
                            "pageindex_rank": rank,
                        },
                    }
                )
    return results


def _retrieve_document(client: PageIndexClient, document_id: str, query: str, source_file: str) -> list[dict]:
    submission = client.submit_query(document_id, query)
    retrieval_id = submission.get("retrieval_id") or submission["id"]
    deadline = time.monotonic() + PER_DOCUMENT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        status = str(retrieval.get("status", "")).lower()
        if status in {"completed", "succeeded"}:
            return _normalize_completed_results(retrieval, document_id, source_file)
        if status in {"failed", "cancelled", "error", "timed_out", "timeout"}:
            return []
        # The installed SDK does not document status-less retrieval completion;
        # partial nodes in an unknown/processing response must not be returned.
        time.sleep(POLL_INTERVAL_SECONDS)
    return []


def _global_rank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Locally rank PageIndex passages across PDFs; local PageIndex rank is a tie-breaker."""
    if not candidates or top_k <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    query_terms = set(tokens)
    scores = BM25Okapi([_tokenize(item["content"]) for item in candidates]).get_scores(tokens)
    ranked = []
    for candidate, score in zip(candidates, scores):
        result = dict(candidate)
        result["metadata"] = dict(candidate["metadata"])
        result["metadata"]["lexical_coverage"] = len(query_terms.intersection(_tokenize(candidate["content"]))) / len(query_terms)
        result["score"] = float(score)
        result["score_type"] = "pageindex_global_bm25"
        result["raw_scores"] = {
            "pageindex": {"score": float(score), "score_type": "pageindex_global_bm25"}
        }
        ranked.append(result)
    return sorted(
        ranked,
        key=lambda item: (
            -item["metadata"]["lexical_coverage"],
            -item["score"],
            item["metadata"]["pageindex_rank"],
            item["metadata"]["document_id"],
        ),
    )[:top_k]


def upload_documents() -> dict[str, str]:
    """Upload uncached landing PDFs and persist their document IDs locally."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Set PAGEINDEX_API_KEY before uploading documents")
    document_ids = _cached_document_ids()
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    for pdf_path in sorted(LANDING_DIR.rglob("*.pdf")):
        key = pdf_path.relative_to(LANDING_DIR).as_posix()
        if key not in document_ids:
            response = client.submit_document(str(pdf_path))
            document_ids[key] = response.get("doc_id") or response["id"]
    DOCUMENT_IDS_PATH.write_text(json.dumps(document_ids, indent=2), encoding="utf-8")
    return document_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query configured PDFs and return globally reranked, citation-ready contexts."""
    if not isinstance(query, str) or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []
    if not query.strip() or not PAGEINDEX_API_KEY:
        return []
    document_ids = _configured_document_ids()
    if not document_ids:
        return []

    source_files = {document_id: name for name, document_id in _cached_document_ids().items()}
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    candidates = []
    for document_id in document_ids:
        try:
            candidates.extend(_retrieve_document(client, document_id, query, source_files.get(document_id, document_id)))
        except Exception:
            # One failed document must not discard completed results from others.
            continue
        if len(candidates) >= MAX_GLOBAL_CANDIDATES:
            break
    return _global_rank(query, candidates[:MAX_GLOBAL_CANDIDATES], top_k)
