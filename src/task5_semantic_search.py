"""Task 5: dense semantic search using cosine similarity."""

from functools import lru_cache
from pathlib import Path

# Must match Task 4. Keeping these lightweight constants here avoids loading the
# large torch/SentenceTransformer stack merely to import this search module.
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load the same embedding model used to build the Task 4 index once."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is missing. Run: pip install sentence-transformers"
        ) from exc
    # SentenceTransformer automatically uses CUDA when available and CPU otherwise.
    return SentenceTransformer(EMBEDDING_MODEL)


def _get_collection():
    """Open the persistent cosine collection created by Task 4."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is missing. Run: pip install chromadb") from exc

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "The vector index does not exist. Run: python src/task4_chunking_indexing.py"
        ) from exc


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return the top chunks ranked by cosine similarity to ``query``.

    Chroma stores cosine *distance* for this collection. For normalized vectors:
    ``cosine_similarity = 1 - cosine_distance``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # An absent directory simply means Task 4 has not indexed anything yet. Returning
    # an empty result makes this retrieval function safe for pipeline fallback logic.
    if not CHROMA_DIR.exists():
        return []

    collection = _get_collection()
    available = collection.count()
    if available == 0:
        return []

    model = _get_embedding_model()
    query_vector = model.encode(
        query.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, available),
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]
    distances = (raw_results.get("distances") or [[]])[0]

    results = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # Do not clamp: true cosine similarity can range from -1 to 1.
        similarity = 1.0 - float(distance)
        results.append({
            "content": document,
            "score": round(similarity, 6),
            "metadata": metadata or {},
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    query = "quy định trả hàng hoàn tiền Shopee"
    matches = semantic_search(query, top_k=5)
    if not matches:
        print("Không có index. Hãy chạy Task 4 trước.")
    for index, match in enumerate(matches, 1):
        source = match["metadata"].get("source", "unknown")
        print(f"{index}. [{match['score']:.4f}] {source}")
        print(f"   {match['content'][:160]}...")
