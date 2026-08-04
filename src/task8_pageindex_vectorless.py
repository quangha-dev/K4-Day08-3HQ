"""Task 8 — bounded PageIndex fallback with local global reranking."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from pageindex import PageIndexAPIError, PageIndexClient

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


def _first_not_none(item: dict, *keys: str):
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


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
                page = _first_not_none(item, "page", "page_number", "page_label")
                section = _first_not_none(item, "section_title", "title")
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
                            "section": "" if section is None else section,
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
    """Globally rank PageIndex passages by local BM25, not PageIndex rank."""
    if not candidates or top_k <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    query_terms = set(tokens)
    bm25_scores = BM25Okapi([_tokenize(item["content"]) for item in candidates]).get_scores(tokens)
    ranked = []
    for candidate, bm25_score in zip(candidates, bm25_scores):
        result = dict(candidate)
        result["metadata"] = dict(candidate["metadata"])
        coverage = len(query_terms.intersection(_tokenize(candidate["content"]))) / len(query_terms)
        result["metadata"]["lexical_coverage"] = coverage
        result["score"] = float(bm25_score)
        result["score_type"] = "pageindex_global_bm25"
        result["raw_scores"] = {
            "pageindex": {"score": float(bm25_score), "score_type": "pageindex_global_bm25"},
            "lexical_coverage": {"score": coverage, "score_type": "fraction"},
        }
        ranked.append(result)
    return sorted(
        ranked,
        key=lambda item: (
            -item["score"],
            -item["metadata"]["lexical_coverage"],
            item["metadata"]["pageindex_rank"],
            item["metadata"]["document_id"],
        ),
    )[:top_k]


def _candidate_key(candidate: dict) -> tuple:
    metadata = candidate["metadata"]
    return (metadata["document_id"], metadata["pageindex_rank"], candidate["content"])


def _fairly_bound_candidates(query: str, per_document: list[list[dict]]) -> list[dict]:
    """Let each PDF nominate one result before filling spare global capacity round-robin."""
    limit = MAX_GLOBAL_CANDIDATES
    primaries = [candidates[0] for candidates in per_document if candidates]
    if len(primaries) > limit:
        return _global_rank(query, primaries, limit)

    selected = _global_rank(query, primaries, len(primaries))
    selected_keys = {_candidate_key(candidate) for candidate in selected}
    for rank_index in range(1, max((len(candidates) for candidates in per_document), default=0)):
        for candidates in per_document:
            if len(selected) >= limit:
                return selected
            if rank_index >= len(candidates):
                continue
            candidate = candidates[rank_index]
            if _candidate_key(candidate) not in selected_keys:
                selected.append(candidate)
                selected_keys.add(_candidate_key(candidate))
    return selected


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
    per_document = []
    for document_id in document_ids:
        try:
            per_document.append(
                _retrieve_document(client, document_id, query, source_files.get(document_id, document_id))
            )
        except PageIndexAPIError:
            # One API-failed document must not discard completed results from others.
            per_document.append([])
            continue
    candidates = _fairly_bound_candidates(query, per_document)
    return _global_rank(query, candidates, top_k)
