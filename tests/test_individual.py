"""
Day 8 v2 — RAG Pipeline
Automated Test Suite cho bài cá nhân.

Chạy:
    pytest tests/test_individual.py -v

Mỗi task được test riêng. Tổng: 50 điểm.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Project root
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"

# Add src to path
sys.path.insert(0, str(PROJECT_DIR))


# ===========================================================================
# Task 1 — Thu thập văn bản pháp luật (3 điểm)
# ===========================================================================

class TestTask1(unittest.TestCase):
    """Task 1: Thu thập ≥3 văn bản pháp luật vào data/landing/legal/"""

    def test_landing_legal_dir_exists(self):
        """data/landing/legal/ tồn tại."""
        legal_dir = DATA_DIR / "landing" / "legal"
        self.assertTrue(legal_dir.exists(), f"Thư mục không tồn tại: {legal_dir}")

    def test_minimum_3_legal_files(self):
        """Có tối thiểu 3 file PDF/DOCX trong data/landing/legal/"""
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")

        valid_extensions = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        self.assertGreaterEqual(
            len(files), 3,
            f"Cần tối thiểu 3 file pháp luật, hiện có {len(files)}: {[f.name for f in files]}"
        )

    def test_files_not_empty(self):
        """Các file pháp luật không rỗng (>1KB)."""
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")

        valid_extensions = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        for f in files:
            self.assertGreater(
                f.stat().st_size, 1024,
                f"File {f.name} quá nhỏ ({f.stat().st_size} bytes), có thể bị lỗi"
            )


# ===========================================================================
# Task 2 — Crawl bài báo (3 điểm)
# ===========================================================================

class TestTask2(unittest.TestCase):
    """Task 2: Crawl ≥5 bài báo vào data/landing/news/"""

    def test_landing_news_dir_exists(self):
        """data/landing/news/ tồn tại."""
        news_dir = DATA_DIR / "landing" / "news"
        self.assertTrue(news_dir.exists(), f"Thư mục không tồn tại: {news_dir}")

    def test_minimum_5_news_files(self):
        """Có tối thiểu 5 file trong data/landing/news/"""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        valid_extensions = {".json", ".html", ".md", ".txt"}
        files = [f for f in news_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        self.assertGreaterEqual(
            len(files), 5,
            f"Cần tối thiểu 5 bài báo, hiện có {len(files)}"
        )

    def test_news_files_have_content(self):
        """Mỗi file bài báo có nội dung (>500 bytes)."""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        files = [f for f in news_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        for f in files[:5]:  # Check first 5
            self.assertGreater(
                f.stat().st_size, 500,
                f"File {f.name} quá nhỏ, có thể crawl bị lỗi"
            )

    def test_json_files_have_metadata(self):
        """File JSON có các trường metadata cần thiết."""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        json_files = [f for f in news_dir.iterdir() if f.suffix == ".json"]
        if not json_files:
            self.skipTest("Không có file JSON (có thể dùng format khác)")

        for f in json_files[:3]:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertIn("url", data, f"{f.name} thiếu trường 'url'")


# ===========================================================================
# Task 3 — Convert markdown (4 điểm)
# ===========================================================================

class TestTask3(unittest.TestCase):
    """Task 3: Convert toàn bộ files sang markdown trong data/standardized/"""

    def test_standardized_dir_exists(self):
        """data/standardized/ tồn tại."""
        self.assertTrue(
            (DATA_DIR / "standardized").exists(),
            "Thư mục data/standardized/ chưa tồn tại"
        )

    def test_has_markdown_files(self):
        """Có ít nhất 1 file .md trong data/standardized/"""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        md_files = list(std_dir.rglob("*.md"))
        self.assertGreater(len(md_files), 0, "Không tìm thấy file .md nào")

    def test_converted_files_have_content(self):
        """File markdown đã convert có nội dung (>200 chars)."""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        md_files = list(std_dir.rglob("*.md"))
        if not md_files:
            self.skipTest("Chưa có file markdown")

        for f in md_files[:5]:
            content = f.read_text(encoding="utf-8")
            self.assertGreater(
                len(content), 200,
                f"{f.name} quá ngắn ({len(content)} chars), convert có thể bị lỗi"
            )

    def test_legal_and_news_both_converted(self):
        """Cả legal và news đều được convert."""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        has_legal = (std_dir / "legal").exists() and list((std_dir / "legal").rglob("*.md"))
        has_news = (std_dir / "news").exists() and list((std_dir / "news").rglob("*.md"))
        self.assertTrue(
            has_legal or has_news,
            "Cần ít nhất 1 trong 2 thư mục legal/ hoặc news/ có file .md"
        )


# ===========================================================================
# Task 4 — Chunking & Indexing (7 điểm)
# ===========================================================================

class TestTask4(unittest.TestCase):
    """Task 4: Chunking + Indexing hoạt động."""

    def _import_task4(self):
        try:
            from src.task4_chunking_indexing import (
                load_documents, chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
            )
            return load_documents, chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 4 chưa implement: {e}")

    def test_config_documented(self):
        """CHUNK_SIZE và CHUNK_OVERLAP được cấu hình."""
        _, _, chunk_size, chunk_overlap = self._import_task4()
        self.assertGreater(chunk_size, 0, "CHUNK_SIZE phải > 0")
        self.assertGreater(chunk_overlap, 0, "CHUNK_OVERLAP phải > 0")
        self.assertLess(chunk_overlap, chunk_size, "OVERLAP phải < SIZE")

    def test_load_documents_returns_list(self):
        """load_documents() trả về list of dicts."""
        load_documents, _, _, _ = self._import_task4()
        try:
            docs = load_documents()
            self.assertIsInstance(docs, list)
            if docs:
                self.assertIn("content", docs[0])
        except NotImplementedError:
            self.skipTest("load_documents chưa implement")

    def test_chunk_documents_produces_chunks(self):
        """chunk_documents() tạo ra chunks từ documents."""
        load_documents, chunk_documents, _, _ = self._import_task4()
        try:
            docs = load_documents()
            if not docs:
                self.skipTest("Không có documents để chunk")
            chunks = chunk_documents(docs[:1])  # Test with 1 doc
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0, "Không tạo được chunk nào")
            self.assertIn("content", chunks[0])
        except NotImplementedError:
            self.skipTest("chunk_documents chưa implement")

    def test_chunks_respect_size_limit(self):
        """Mỗi chunk không vượt quá CHUNK_SIZE (+ tolerance 10%)."""
        load_documents, chunk_documents, chunk_size, _ = self._import_task4()
        try:
            docs = load_documents()
            if not docs:
                self.skipTest("Không có documents")
            chunks = chunk_documents(docs[:1])
            max_allowed = int(chunk_size * 1.1)
            for i, c in enumerate(chunks[:20]):
                self.assertLessEqual(
                    len(c["content"]), max_allowed,
                    f"Chunk {i} vượt quá size limit: {len(c['content'])} > {max_allowed}"
                )
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 5 — Semantic Search (6 điểm)
# ===========================================================================

class TestTask5(unittest.TestCase):
    """Task 5: Semantic search module."""

    def _import_task5(self):
        try:
            from src.task5_semantic_search import semantic_search
            return semantic_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 5 chưa implement: {e}")

    def test_returns_list(self):
        """semantic_search() trả về list."""
        search = self._import_task5()
        try:
            results = search("payment methods", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("semantic_search chưa implement")

    def test_results_have_required_keys(self):
        """Mỗi result có 'content', 'score', 'metadata'."""
        search = self._import_task5()
        try:
            results = search("return refund policy", top_k=3)
            if not results:
                self.skipTest("Không có kết quả (có thể chưa index)")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_results_sorted_descending(self):
        """Kết quả sorted theo score descending."""
        search = self._import_task5()
        try:
            results = search("ecommerce return policy", top_k=5)
            if len(results) < 2:
                self.skipTest("Không đủ kết quả để test sort")
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_respects_top_k(self):
        """Không trả về nhiều hơn top_k results."""
        search = self._import_task5()
        try:
            results = search("test query", top_k=2)
            self.assertLessEqual(len(results), 2)
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 6 — Lexical Search / BM25 (6 điểm)
# ===========================================================================

class TestTask6(unittest.TestCase):
    """Task 6: Lexical search (BM25)."""

    def _import_task6(self):
        try:
            from src.task6_lexical_search import lexical_search
            return lexical_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 6 chưa implement: {e}")

    def test_returns_list(self):
        """lexical_search() trả về list."""
        search = self._import_task6()
        try:
            results = search("return refund evidence policy", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("lexical_search chưa implement")

    def test_results_have_required_keys(self):
        """Mỗi result có 'content', 'score'."""
        search = self._import_task6()
        try:
            results = search("seller listing regulations", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_results_sorted_descending(self):
        """Kết quả sorted theo BM25 score descending."""
        search = self._import_task6()
        try:
            results = search("order tracking guide", top_k=5)
            if len(results) < 2:
                self.skipTest("Không đủ kết quả")
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_keyword_match_scores_higher(self):
        """Query có keyword match phải có score > 0."""
        search = self._import_task6()
        try:
            results = search("payment methods", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            # Ít nhất 1 result phải có score > 0
            max_score = max(r["score"] for r in results)
            self.assertGreater(max_score, 0, "Tất cả score = 0, BM25 có thể bị lỗi")
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_bm25_prioritizes_the_document_with_the_exact_keyword(self):
        """BM25 ưu tiên document có từ khóa truy vấn."""
        from src import task6_lexical_search

        corpus = [
            {"content": "Return and refund policy", "metadata": {"source": "policy"}},
            {"content": "Payment methods", "metadata": {"source": "payments"}},
            {"content": "Seller listing regulations", "metadata": {"source": "listing"}},
        ]
        with patch.object(task6_lexical_search, "CORPUS", corpus):
            results = task6_lexical_search.lexical_search("refund", top_k=1)

        self.assertEqual(results[0]["metadata"]["source"], "policy")
        self.assertGreater(results[0]["score"], 0)

    def test_bm25_returns_no_results_when_no_terms_match(self):
        """Query không có keyword không được trả document score 0."""
        from src import task6_lexical_search

        corpus = [
            {"content": "Return and refund policy", "metadata": {}},
            {"content": "Payment methods", "metadata": {}},
            {"content": "Seller listing regulations", "metadata": {}},
        ]
        with patch.object(task6_lexical_search, "CORPUS", corpus):
            self.assertEqual(task6_lexical_search.lexical_search("zzzz-no-match"), [])

    def test_empty_corpus_is_reloaded_after_markdown_is_added(self):
        """A process started before Task 3 can search documents added later."""
        from src import task6_lexical_search

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            with (
                patch.object(task6_lexical_search, "CORPUS", None),
                patch.object(task6_lexical_search, "STANDARDIZED_DIR", corpus_dir),
                patch.object(task6_lexical_search, "_CACHED_CORPUS", None),
                patch.object(task6_lexical_search, "_CACHED_BM25", None),
            ):
                self.assertEqual(task6_lexical_search.lexical_search("refund"), [])
                for index, text in enumerate(("Refund policy", "Payment methods", "Listing rules")):
                    (corpus_dir / f"doc-{index}.md").write_text(text, encoding="utf-8")

                results = task6_lexical_search.lexical_search("refund")

        self.assertEqual(results[0]["content"], "Refund policy")


# ===========================================================================
# Task 7 — Reranking (6 điểm)
# ===========================================================================

class TestTask7(unittest.TestCase):
    """Task 7: Reranking module."""

    def _import_task7(self):
        try:
            from src.task7_reranking import rerank
            return rerank
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 7 chưa implement: {e}")

    def test_rerank_returns_list(self):
        """rerank() trả về list."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": "Payment methods overview", "score": 0.8, "metadata": {}},
            {"content": "Seller listing regulations", "score": 0.6, "metadata": {}},
            {"content": "Python programming", "score": 0.4, "metadata": {}},
        ]
        try:
            results = rerank_fn("payment methods", candidates, top_k=2)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("rerank chưa implement")

    def test_rerank_respects_top_k(self):
        """Rerank trả về đúng top_k results."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": f"Document {i}", "score": 0.9 - i * 0.1, "metadata": {}}
            for i in range(10)
        ]
        try:
            results = rerank_fn("test query", candidates, top_k=3)
            self.assertLessEqual(len(results), 3)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_rerank_has_score(self):
        """Kết quả rerank có trường 'score'."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": "Return and refund policy", "score": 0.7, "metadata": {}},
            {"content": "Order tracking guide", "score": 0.5, "metadata": {}},
        ]
        try:
            results = rerank_fn("return policy", candidates, top_k=2)
            if results:
                self.assertIn("score", results[0])
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_rrf_rewards_a_document_found_by_both_rankers(self):
        """Document xuất hiện ở cả dense và lexical lists phải đứng đầu."""
        from src.task7_reranking import rerank_rrf

        shared = {"content": "Refund policy", "score": 0.5, "metadata": {}}
        results = rerank_rrf(
            [
                [shared, {"content": "Shipping guide", "score": 0.4, "metadata": {}}],
                [{"content": "Payment methods", "score": 2.0, "metadata": {}}, shared],
            ],
            top_k=1,
        )

        self.assertEqual(results[0]["content"], "Refund policy")
        self.assertAlmostEqual(results[0]["score"], 1 / 61 + 1 / 62)


# ===========================================================================
# Task 8 — PageIndex Vectorless (4 điểm)
# ===========================================================================

class TestTask8(unittest.TestCase):
    """Task 8: PageIndex vectorless RAG."""

    def _import_task8(self):
        try:
            from src.task8_pageindex_vectorless import pageindex_search
            return pageindex_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 8 chưa implement: {e}")

    def test_function_exists(self):
        """pageindex_search() function tồn tại."""
        search = self._import_task8()
        self.assertTrue(callable(search))

    def test_returns_list_with_source_marker(self):
        """Kết quả có 'source': 'pageindex'."""
        search = self._import_task8()
        try:
            results = search("payment methods", top_k=2)
            self.assertIsInstance(results, list)
            if results:
                self.assertEqual(results[0].get("source"), "pageindex")
        except (NotImplementedError, Exception) as e:
            self.skipTest(f"PageIndex chưa sẵn sàng: {e}")

    def test_parses_pageindex_results_without_network_access(self):
        """PageIndex results được đổi về retrieval contract chung."""
        from src import task8_pageindex_vectorless

        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def submit_query(self, doc_id, query):
                return {"retrieval_id": f"retrieval-{doc_id}"}

            def get_retrieval(self, retrieval_id):
                return {
                    "status": "completed",
                    "retrieved_nodes": [
                        {
                            "relevant_contents": [
                                [{"section_title": "Refunds", "relevant_content": "Refund within 15 days."}]
                            ]
                        }
                    ]
                }

        with (
            patch.object(task8_pageindex_vectorless, "PAGEINDEX_API_KEY", "test-key"),
            patch.object(task8_pageindex_vectorless, "PAGEINDEX_DOCUMENT_IDS", ("doc-1",)),
            patch.object(task8_pageindex_vectorless, "PageIndexClient", FakeClient),
        ):
            results = task8_pageindex_vectorless.pageindex_search("refund", top_k=1)

        self.assertEqual(results[0]["content"], "Refund within 15 days.")
        self.assertEqual(results[0]["source"], "pageindex")


# ===========================================================================
# Task 9 — Retrieval Pipeline (7 điểm)
# ===========================================================================

class TestTask9(unittest.TestCase):
    """Task 9: Retrieval pipeline hoàn chỉnh."""

    def _import_task9(self):
        try:
            from src.task9_retrieval_pipeline import retrieve
            return retrieve
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 9 chưa implement: {e}")

    def test_retrieve_returns_list(self):
        """retrieve() trả về list of dicts."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("return refund policy", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("retrieve chưa implement")

    def test_results_have_required_keys(self):
        """Kết quả có 'content', 'score', 'source'."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("ecommerce return policy", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
                self.assertIn("source", r)
                self.assertIn(r["source"], ["hybrid", "pageindex"])
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_respects_top_k(self):
        """Không trả về nhiều hơn top_k."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("test", top_k=2)
            self.assertLessEqual(len(results), 2)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_fallback_logic_exists(self):
        """Pipeline có fallback logic (không crash khi hybrid trả rỗng)."""
        retrieve_fn = self._import_task9()
        try:
            # Query rất obscure → hybrid có thể không tìm thấy → fallback
            results = retrieve_fn("xyzabc123nonsense", top_k=3, score_threshold=0.99)
            # Không crash = pass
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Retrieval regression coverage — Tasks 6–9
# ===========================================================================

class TestRetrievalRegressions(unittest.TestCase):
    def test_bm25_keeps_matching_zero_and_negative_scores(self):
        from src import task6_lexical_search as task6

        class Scores:
            def get_scores(self, tokens):
                return [0.0, -0.5, 0.0]

        corpus = [
            {"content": "refund policy", "metadata": {}},
            {"content": "refund guide", "metadata": {}},
            {"content": "payment methods", "metadata": {}},
        ]
        with (
            patch.object(task6, "CORPUS", corpus),
            patch.object(task6, "build_bm25_index", return_value=Scores()),
        ):
            results = task6.lexical_search("refund", top_k=3)

        self.assertEqual([result["content"] for result in results], ["refund policy", "refund guide"])
        self.assertEqual([result["score"] for result in results], [0.0, -0.5])

    def test_bm25_rejects_no_overlap_and_invalid_input(self):
        from src import task6_lexical_search as task6

        corpus = [{"content": "refund policy", "metadata": {}}]
        with patch.object(task6, "CORPUS", corpus):
            self.assertEqual(task6.lexical_search("shipping"), [])
            self.assertEqual(task6.lexical_search("", top_k=1), [])
            self.assertEqual(task6.lexical_search("refund", top_k=0), [])
            self.assertEqual(task6.lexical_search("refund", top_k="bad"), [])

    def test_markdown_chunker_keeps_heading_context_and_stable_ids(self):
        from src import task6_lexical_search as task6

        markdown = "# Returns\n\nOverview paragraph.\n\n## Refunds\n\nRefund within 15 days."
        chunks = task6._markdown_chunks(markdown, "legal/policy.md")

        self.assertIn("# Returns", chunks[0]["content"])
        self.assertIn("## Refunds", chunks[1]["content"])
        self.assertIn("# Returns", chunks[1]["content"])
        self.assertEqual([chunk["metadata"]["chunk_id"] for chunk in chunks], [
            "legal/policy.md#chunk-0", "legal/policy.md#chunk-1"
        ])

    def test_markdown_chunker_splits_oversized_sections_without_dropping_text(self):
        from src import task6_lexical_search as task6

        body = "word " * (task6.MAX_CHUNK_CHARS // 2)
        chunks = task6._markdown_chunks(f"# Policy\n\n{body}", "news/policy.md")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("# Policy" in chunk["content"] for chunk in chunks))
        self.assertEqual("".join(chunk["content"].replace("# Policy", "") for chunk in chunks).replace("\n", "").replace(" ", ""), body.replace(" ", ""))

    def test_markdown_chunk_ids_distinguish_repeated_filenames(self):
        from src import task6_lexical_search as task6

        left = task6._markdown_chunks("# A\n\nContent", "legal/policy.md")[0]
        right = task6._markdown_chunks("# B\n\nContent", "news/policy.md")[0]
        self.assertNotEqual(left["metadata"]["chunk_id"], right["metadata"]["chunk_id"])

    def test_bm25_override_cache_reuses_and_invalidates_index(self):
        from src import task6_lexical_search as task6

        corpus = [
            {"content": "refund policy", "metadata": {}},
            {"content": "payment methods", "metadata": {}},
            {"content": "listing rules", "metadata": {}},
        ]
        original = task6.build_bm25_index
        with (
            patch.object(task6, "CORPUS", corpus),
            patch.object(task6, "build_bm25_index", wraps=original) as build,
        ):
            task6.invalidate_bm25_cache()
            task6.lexical_search("refund")
            task6.lexical_search("refund")
            self.assertEqual(build.call_count, 1)
            task6.invalidate_bm25_cache()
            task6.lexical_search("refund")
            self.assertEqual(build.call_count, 2)

    def test_rrf_keeps_empty_candidates_separate_and_tracks_sources(self):
        from src.task7_reranking import rerank_rrf

        results = rerank_rrf([
            [
                {"content": "", "score": 0.81, "score_type": "cosine", "retrieval_source": "dense", "metadata": {}},
                {"content": "", "score": 0.42, "score_type": "cosine", "retrieval_source": "dense", "metadata": {}},
            ],
            [{"content": "", "score": 5.42, "score_type": "cosine", "retrieval_source": "sparse", "metadata": {}}],
        ], top_k=3)

        self.assertEqual(len(results), 3)
        self.assertIn("dense", results[0]["raw_scores"])
        self.assertEqual(results[0]["raw_scores"]["dense"]["score_type"], "cosine")
        sparse_result = next(result for result in results if "sparse" in result["raw_scores"])
        self.assertEqual(sparse_result["raw_scores"]["sparse"]["score"], 5.42)

    def test_rerank_rejects_unimplemented_methods(self):
        from src.task7_reranking import rerank

        with self.assertRaisesRegex(ValueError, "Only RRF"):
            rerank("refund", [], method="mmr")

    def test_pageindex_ignores_partial_results_then_normalizes_completed_results(self):
        from src import task8_pageindex_vectorless as task8

        class FakeClient:
            def submit_query(self, doc_id, query):
                return {"retrieval_id": doc_id}

            def get_retrieval(self, retrieval_id):
                if not hasattr(self, "calls"):
                    self.calls = 0
                self.calls += 1
                if self.calls == 1:
                    return {"status": "processing", "retrieved_nodes": [{"relevant_contents": [[{"relevant_content": "partial"}]]}]}
                return {"status": "completed", "retrieved_nodes": [{"relevant_contents": [[{"section_title": "Refunds", "relevant_content": "Refund policy", "page": 4}]]}]}

        with (
            patch.object(task8, "PAGEINDEX_API_KEY", "test-key"),
            patch.object(task8, "PAGEINDEX_DOCUMENT_IDS", ("doc-1",)),
            patch.object(task8, "PageIndexClient", lambda api_key: FakeClient()),
            patch.object(task8, "POLL_INTERVAL_SECONDS", 0),
        ):
            results = task8.pageindex_search("refund", top_k=1)

        self.assertEqual(results[0]["content"], "Refund policy")
        self.assertEqual(results[0]["metadata"]["document_id"], "doc-1")
        self.assertEqual(results[0]["metadata"]["page"], 4)
        self.assertEqual(results[0]["metadata"]["source_file"], "doc-1")

    def test_pageindex_ranks_relevant_passage_across_documents_and_survives_failures(self):
        from src import task8_pageindex_vectorless as task8

        class FakeClient:
            def submit_query(self, doc_id, query):
                return {"retrieval_id": doc_id}

            def get_retrieval(self, retrieval_id):
                if retrieval_id == "bad":
                    return {"status": "failed"}
                content = "General policy information" if retrieval_id == "doc-a" else "Refund policy requirements"
                return {"status": "succeeded", "retrieved_nodes": [{"relevant_contents": [[{"section_title": "Policy", "relevant_content": content}]]}]}

        with (
            patch.object(task8, "PAGEINDEX_API_KEY", "test-key"),
            patch.object(task8, "PAGEINDEX_DOCUMENT_IDS", ("doc-a", "bad", "doc-b")),
            patch.object(task8, "PageIndexClient", lambda api_key: FakeClient()),
        ):
            results = task8.pageindex_search("refund policy", top_k=2)

        self.assertEqual(results[0]["content"], "Refund policy requirements")
        self.assertEqual(results[0]["score_type"], "pageindex_global_bm25")
        self.assertEqual({result["metadata"]["document_id"] for result in results}, {"doc-a", "doc-b"})

    def test_pageindex_timeout_returns_no_results(self):
        from src import task8_pageindex_vectorless as task8

        class FakeClient:
            def submit_query(self, doc_id, query):
                return {"retrieval_id": doc_id}

            def get_retrieval(self, retrieval_id):
                return {"status": "processing"}

        with (
            patch.object(task8, "PAGEINDEX_API_KEY", "test-key"),
            patch.object(task8, "PAGEINDEX_DOCUMENT_IDS", ("doc-1",)),
            patch.object(task8, "PageIndexClient", lambda api_key: FakeClient()),
            patch.object(task8, "PER_DOCUMENT_TIMEOUT_SECONDS", 0),
        ):
            self.assertEqual(task8.pageindex_search("refund"), [])

    def test_retrieve_uses_dense_cosine_only_for_fallback_and_preserves_schema(self):
        from src import task9_retrieval_pipeline as task9

        dense = [{"content": "dense refund", "score": 0.2, "score_type": "cosine", "metadata": {"chunk_id": "d1"}}]
        sparse = [{"content": "sparse refund", "score": 999.0, "score_type": "bm25", "metadata": {"chunk_id": "s1"}}]
        fallback = [{"content": "PageIndex refund", "score": 3.0, "score_type": "pageindex_global_bm25", "metadata": {}, "source": "pageindex"}]
        with (
            patch.object(task9, "semantic_search", return_value=dense),
            patch.object(task9, "lexical_search", return_value=sparse),
            patch.object(task9, "pageindex_search", return_value=fallback) as pageindex,
        ):
            results = task9.retrieve("refund", score_threshold=0.3)

        pageindex.assert_called_once()
        self.assertEqual(results[0]["retrieval_source"], "pageindex")
        self.assertEqual(results[0]["metadata"]["page"], None)
        self.assertEqual(results[0]["raw_scores"]["pageindex"]["score_type"], "pageindex_global_bm25")

    def test_retrieve_does_not_fallback_for_high_dense_cosine_despite_low_rrf(self):
        from src import task9_retrieval_pipeline as task9

        dense = [{"content": "dense refund", "score": 0.9, "score_type": "cosine", "metadata": {"chunk_id": "d1"}}]
        sparse = [{"content": "sparse refund", "score": 1.0, "score_type": "bm25", "metadata": {"chunk_id": "s1"}}]
        with (
            patch.object(task9, "semantic_search", return_value=dense),
            patch.object(task9, "lexical_search", return_value=sparse),
            patch.object(task9, "pageindex_search", return_value=[]) as pageindex,
        ):
            results = task9.retrieve("refund", score_threshold=0.3)

        pageindex.assert_not_called()
        self.assertEqual(results[0]["score_type"], "rrf")
        self.assertIn("dense", results[0]["raw_scores"])

    def test_retrieve_falls_back_when_dense_results_are_empty(self):
        from src import task9_retrieval_pipeline as task9

        fallback = [{"content": "PageIndex refund", "score": 1.0, "score_type": "pageindex_global_bm25", "metadata": {}, "source": "pageindex"}]
        with (
            patch.object(task9, "semantic_search", return_value=[]),
            patch.object(task9, "lexical_search", return_value=[]),
            patch.object(task9, "pageindex_search", return_value=fallback) as pageindex,
        ):
            results = task9.retrieve("refund")

        pageindex.assert_called_once()
        self.assertEqual(results[0]["retrieval_source"], "pageindex")

    def test_retrieve_degrades_safely_when_dense_retrieval_is_unimplemented(self):
        from src import task9_retrieval_pipeline as task9

        sparse = [{"content": "Refund policy", "score": 2.0, "score_type": "bm25", "metadata": {"chunk_id": "s1"}}]
        with (
            patch.object(task9, "semantic_search", side_effect=NotImplementedError),
            patch.object(task9, "lexical_search", return_value=sparse),
            patch.object(task9, "pageindex_search", return_value=[]),
        ):
            results = task9.retrieve("refund")

        self.assertEqual(results[0]["retrieval_source"], "hybrid")


# ===========================================================================
# Task 10 — Generation có Citation (4 điểm)
# ===========================================================================

class TestTask10(unittest.TestCase):
    """Task 10: Generation có citation + document reordering."""

    def _import_task10(self):
        try:
            from src.task10_generation import (
                generate_with_citation, reorder_for_llm, format_context
            )
            return generate_with_citation, reorder_for_llm, format_context
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 10 chưa implement: {e}")

    def test_reorder_function_exists(self):
        """reorder_for_llm() function hoạt động."""
        _, reorder, _ = self._import_task10()
        chunks = [
            {"content": f"Chunk {i}", "score": 1.0 - i * 0.1}
            for i in range(5)
        ]
        try:
            reordered = reorder(chunks)
            self.assertEqual(len(reordered), 5, "Reorder phải giữ nguyên số lượng chunks")
            # Chunk đầu tiên (important nhất) vẫn ở đầu
            self.assertEqual(reordered[0]["content"], "Chunk 0")
        except NotImplementedError:
            self.skipTest("reorder_for_llm chưa implement")

    def test_format_context_includes_source(self):
        """format_context() có thông tin source cho citation."""
        _, _, format_ctx = self._import_task10()
        chunks = [
            {"content": "Nội dung pháp luật", "score": 0.9,
             "metadata": {"source": "luat-phong-chong-ma-tuy.pdf", "type": "legal"}}
        ]
        try:
            ctx = format_ctx(chunks)
            self.assertIn("luat-phong-chong-ma-tuy", ctx)
        except NotImplementedError:
            self.skipTest("format_context chưa implement")

    def test_generate_returns_dict_with_answer(self):
        """generate_with_citation() trả về dict có 'answer'."""
        generate, _, _ = self._import_task10()
        try:
            result = generate("What payment methods does Shopee support?")
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            self.assertGreater(len(result["answer"]), 0)
        except NotImplementedError:
            self.skipTest("generate_with_citation chưa implement")
        except Exception as e:
            # API key missing, etc — still check structure exists
            self.skipTest(f"Generation error (có thể thiếu API key): {e}")


# ===========================================================================
# Summary
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
