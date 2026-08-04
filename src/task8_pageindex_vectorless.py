"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
DOCUMENT_IDS_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
PAGEINDEX_DOCUMENT_IDS = tuple(
    doc_id.strip() for doc_id in os.getenv("PAGEINDEX_DOCUMENT_IDS", "").split(",") if doc_id.strip()
)


def _cached_document_ids() -> dict[str, str]:
    if not DOCUMENT_IDS_PATH.exists():
        return {}
    try:
        return json.loads(DOCUMENT_IDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _configured_document_ids() -> tuple[str, ...]:
    if PAGEINDEX_DOCUMENT_IDS:
        return PAGEINDEX_DOCUMENT_IDS
    return tuple(_cached_document_ids().values())


def _extract_results(retrieval: dict, top_k: int, document_id: str) -> list[dict]:
    results = []
    for node in retrieval.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []):
            for item in group:
                content = item.get("relevant_content", "").strip()
                if content:
                    rank_position = len(results) + 1
                    results.append(
                        {
                            "content": content,
                            "score": 1.0 / rank_position,
                            "score_type": "pageindex_rank",
                            "metadata": {
                                "document_id": document_id,
                                "section": item.get("section_title", ""),
                                "rank_position": rank_position,
                            },
                            "source": "pageindex",
                        }
                    )
                    if len(results) == top_k:
                        return results
    return results


def upload_documents():
    """
    Upload landing PDF documents lên PageIndex và cache document IDs.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Set PAGEINDEX_API_KEY before uploading documents")

    document_ids = _cached_document_ids()
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    for pdf_path in LANDING_DIR.rglob("*.pdf"):
        key = str(pdf_path.relative_to(LANDING_DIR))
        if key not in document_ids:
            response = client.submit_document(str(pdf_path))
            document_ids[key] = response.get("doc_id") or response["id"]
    DOCUMENT_IDS_PATH.write_text(json.dumps(document_ids, indent=2), encoding="utf-8")
    return document_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'score_type': 'pageindex_rank',
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    document_ids = _configured_document_ids()
    if top_k <= 0 or not PAGEINDEX_API_KEY or not document_ids:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    all_results = []
    for document_id in document_ids:
        try:
            response = client.submit_query(document_id, query)
            retrieval_id = response.get("retrieval_id") or response["id"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "").lower()
                if status == "completed" or (status not in ("failed", "cancelled", "error") and "retrieved_nodes" in retrieval):
                    results = _extract_results(retrieval, top_k, document_id)
                    if results:
                        all_results.extend(results)
                    break
                if status in ("failed", "cancelled", "error"):
                    break
                time.sleep(2)
        except Exception as e:
            print(f"⚠ PageIndex error for doc {document_id}: {e}")
            continue

    return sorted(all_results, key=lambda r: r["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
