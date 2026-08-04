# PHÂN CÔNG NHÓM — K4 Day08 RAG Pipeline (5 thành viên)

> Repo: `K4-Day08-3HQ` · Thời lượng: 180 phút · Điểm: 50 cá nhân + 50 nhóm
> Mô hình Git đã chốt: **1 repo chung, mỗi người 1 branch riêng**.

---

## 0a. Trạng thái repo hiện tại (đã kiểm tra thực tế)

| Hạng mục | Trạng thái | Ai lo |
|---|---|---|
| `data/landing/legal/` | ❌ **Rỗng** (chỉ có `.gitkeep`) | TV2 — Task 1 |
| `data/landing/news/` | ❌ **Rỗng** | TV2 — Task 2 |
| `data/standardized/legal|news/` | ❌ **Rỗng** | TV2 — Task 3 |
| Bộ "11 file mẫu có sẵn" đề bài nhắc tới | ❌ **Không tồn tại trong repo** | — |
| `chroma_db/` | ❌ Chưa có (sinh ra khi chạy Task 4) | TV2 |
| `.env` | ❌ Chưa có, mới chỉ có `.env.example` | TV1 |
| `src/task1..10_*.py` | ⚠️ Có file nhưng toàn `TODO`, chưa implement | mỗi chủ file |
| `src/supervisor.py` | ❌ Không có (README có nhắc, tuỳ chọn) | TV1 |
| `app.py` | ⚠️ Có khung UI sidebar + câu hỏi gợi ý, chưa nối pipeline | TV4 |
| `group_project/evaluation/golden_dataset.json` | ⚠️ Mới có **3 câu**, cần ≥15 | TV5 |
| `group_project/evaluation/eval_pipeline.py` | ⚠️ Khung rỗng, toàn `TODO` | TV5 |
| `tests/test_individual.py` | ✅ Đầy đủ 35 test | không ai sửa |

> **Hệ quả quan trọng:** dữ liệu là đường găng. 11/35 test (Task 1–3) phụ thuộc hoàn toàn vào TV2 hoàn thành trong 25 phút của CP1. Nếu TV2 trễ, cả nhóm kẹt. Vì vậy TV1 nên ngồi cạnh hỗ trợ TV2 ở CP1 thay vì chỉ điều phối.

---

## 0. Nguyên tắc vàng để KHÔNG BAO GIỜ conflict

Toàn bộ conflict trong lab này chỉ đến từ 3 nguồn. Chặn cả 3 là xong:

| Nguồn conflict | Cách chặn |
|---|---|
| 2 người sửa cùng 1 file `src/taskN_*.py` | **File ownership tuyệt đối** — mỗi file chỉ có đúng 1 chủ (bảng mục 2). Không phải chủ file thì *tuyệt đối không sửa*, kể cả sửa 1 dòng typo. |
| Nhiều người commit file nhị phân / dữ liệu sinh ra | `chroma_db/`, `.env`, `__pycache__/`, `*.pdf` cache **không được commit**. Bổ sung `.gitignore` ngay ở CP0 (chỉ TV1 làm). |
| Ai cũng phải làm Task 1–10 để đủ 35 test cá nhân | **Tách bạch 2 không gian**: bài cá nhân nằm ở branch riêng `dev/tvX` (không merge vào main); bài nhóm nằm ở branch `feat/*` (được merge vào main). |

### Sơ đồ branch

```
main  ──●────────●──────────●──────────●──────────●───►  (chỉ nhận PR, không ai commit thẳng)
         \        \          \          \          \
          feat/data      feat/dense   feat/sparse   feat/app   feat/eval
          (TV2)          (TV2)        (TV3)         (TV4)      (TV5)

dev/tv1 dev/tv2 dev/tv3 dev/tv4 dev/tv5  ──►  bài CÁ NHÂN, KHÔNG merge vào main
```

**Quy tắc bắt buộc:**

1. Cấm `git push origin main` trực tiếp. Chỉ merge qua Pull Request, TV1 (Leader) là người duy nhất bấm Merge.
2. Trước mỗi lần push: `git pull --rebase origin main` → sửa xong → `git push origin <branch>`.
3. Commit message: `[TVx][TaskN] mô tả ngắn`. Ví dụ: `[TV3][Task7] implement rerank_rrf k=60`.
4. Mỗi branch chỉ đụng đúng các file trong cột "File sở hữu" của mình. Cần sửa file người khác → nhắn trong nhóm chat, chủ file tự sửa.
5. Không ai commit `chroma_db/` (mỗi máy tự build lại bằng `python -m src.task4_chunking_indexing`).

---

## 1. Bản đồ 5 vai trò (Phương án B — Chuyên sâu Retrieval)

| TV | Vai trò | Task chính | Branch |
|---|---|---|---|
| **TV1** | Team Leader & RAG Architect | Task 9 (pipeline) + review + merge + demo | `dev/tv1`, `feat/pipeline` |
| **TV2** | Data & Dense Search Dev | Task 1, 2, 3, 4, 5 | `dev/tv2`, `feat/data`, `feat/dense` |
| **TV3** | Sparse & Reranking Dev | Task 6, 7, 8 | `dev/tv3`, `feat/sparse` |
| **TV4** | Frontend & Chatbot Dev | Task 10 + `app.py` | `dev/tv4`, `feat/app` |
| **TV5** | Evaluation & QA Engineer | golden_dataset + RAGAS + results.md | `dev/tv5`, `feat/eval` |

### Đối chiếu với Phương Án B trong đề bài

| Đề bài (Phương Án B) | File này | Khớp? |
|---|---|---|
| Role 1: `supervisor.py` & Task 9 | TV1: Task 9 + `supervisor.py` (tuỳ chọn) + review/merge/demo | ✅ |
| Role 2: Task 1–3 + Task 4 + Task 5 | TV2: Task 1, 2, 3, 4, 5 | ✅ |
| Role 3: Task 6 + Task 7 + Task 8 | TV3: Task 6, 7, 8 | ✅ |
| Role 4: `app.py` + Task 10 | TV4: Task 10 + `app.py` | ✅ |
| Role 5: `golden_dataset.json` + RAGAS + `results.md` | TV5: y hệt | ✅ |

**2 điểm bổ sung so với đề bài** (đề chỉ nói "ai làm gì", không nói "ai được sửa file nào"):

1. Thêm **bảng sở hữu file** ở mục 2 — đây mới là thứ thực sự chặn conflict khi 5 người cùng push.
2. Thêm **hợp đồng interface** ở mục 3 — chốt shape dữ liệu để 4 người còn lại code song song bằng mock, không phải ngồi chờ TV2 index xong ChromaDB.

> ⚠️ Lưu ý: mục 3 và mục 4 của đề bài **mâu thuẫn nhau** về Task 2. Mục 3 (Phương Án B) giao Task 1–3 cho Role 2; mục 4 (checkpoint CP1) lại giao Task 2 crawl cho Role 3. File này theo **Phương Án B** — TV2 sở hữu toàn bộ `data/`, TV3 chỉ gửi danh sách URL. Lý do: nếu 2 người cùng ghi vào `data/landing/` thì đúng chỗ dễ conflict nhất, mà `src/task2_crawl_news.py` và `src/task3_convert_markdown.py` lại phụ thuộc chặt vào nhau.

---

## 2. BẢNG SỞ HỮU FILE (quan trọng nhất — in ra dán màn hình)

| File / thư mục | Chủ sở hữu duy nhất | Người khác được phép |
|---|---|---|
| `.gitignore`, `.env.example`, `README.md` | TV1 | chỉ đọc |
| `src/task1_collect_legal_docs.py` | TV2 | chỉ đọc |
| `src/task2_crawl_news.py` | TV2 | chỉ đọc |
| `src/task3_convert_markdown.py` | TV2 | chỉ đọc |
| `src/task4_chunking_indexing.py` | TV2 | chỉ đọc |
| `src/task5_semantic_search.py` | TV2 | chỉ đọc |
| `src/task6_lexical_search.py` | TV3 | chỉ đọc |
| `src/task7_reranking.py` | TV3 | chỉ đọc |
| `src/task8_pageindex_vectorless.py` | TV3 | chỉ đọc |
| `src/task9_retrieval_pipeline.py` | **TV1** | chỉ đọc |
| `src/supervisor.py` (tuỳ chọn, chưa có trong repo) | **TV1** | chỉ đọc |
| `src/task10_generation.py` | TV4 | chỉ đọc |
| `app.py` | TV4 | chỉ đọc |
| `group_project/evaluation/golden_dataset.json` | TV5 | chỉ đọc |
| `group_project/evaluation/eval_pipeline.py` | TV5 | chỉ đọc |
| `group_project/evaluation/results.md` | TV5 | chỉ đọc |
| `data/landing/legal/*` | TV2 (đặt tên `legal_01..03_*`) | không ghi đè |
| `data/landing/news/*` | TV2 (đặt tên `news_01..05_*`) | TV3 chỉ gửi danh sách URL cho TV2 |
| `data/standardized/**` | TV2 (sinh ra từ Task 3) | không ghi đè |
| `tests/test_individual.py` | **KHÔNG AI ĐƯỢC SỬA** | chỉ chạy |
| `chroma_db/` | không commit | mỗi máy tự build |
| `PHAN_CONG_NHOM.md` (file này) | TV1 | chỉ đọc |

> ⚠️ `tests/test_individual.py` là barem chấm điểm. Sửa file này = mất điểm cả nhóm.

---

## 2b. ĐỀ BÀI BẮT BUỘC GÌ, TỰ CHỌN GÌ (trích từ barem `README.md`)

### ✅ BẮT BUỘC — làm sai là mất điểm

| Hạng mục | Ràng buộc cứng |
|---|---|
| Dữ liệu | ≥3 file chính sách (PDF/DOCX) + ≥5 bài crawl + convert `.md` giữ nguyên cấu trúc `legal/`, `news/` |
| Metadata | `customer_role` ∈ `buyer`/`seller`/`both` — **riêng của K4 variant**, thiếu là rớt test Task 4 |
| Vector store | **ChromaDB** (đề ghi rõ "Vector Store mặc định của bài lab") |
| Retrieval | Phải có **cả hai**: semantic (Task 5) + lexical (Task 6). Không được chỉ làm 1 |
| Fallback | Ngưỡng phải so với **điểm cosine gốc** từ `semantic_search`, KHÔNG so với điểm RRF đã fuse |
| Task 9 `source` | Giá trị phải là đúng chuỗi `"hybrid"` hoặc `"pageindex"` |
| Task 10 citation | Output phải có citation dạng `[Nguồn, Năm]` |
| Task 10 no-evidence | Không đủ evidence → trả về đúng câu **"I cannot verify this information"** |
| Task 10 reorder | Quan trọng nhất ở **đầu và cuối**, ít quan trọng ở giữa (chống lost-in-the-middle) |
| Golden dataset | **≥15** cặp Q&A `(question, expected_answer, expected_context)` |
| Eval metrics | Đủ **4 trục**: Faithfulness, Answer Relevance, Context Recall, Context Precision |
| A/B testing | ≥2 config khác nhau + phân tích |
| Báo cáo | Bảng điểm + **phân tích worst performers** + đề xuất cải tiến |
| Chatbot | Có citation + hiển thị source documents + **hỗ trợ follow-up (conversation memory)** |

### 🔓 TỰ CHỌN — nhưng phải giải thích lý do trong code comment / demo

| Hạng mục | Các lựa chọn đề bài cho phép | Nhóm mình chọn |
|---|---|---|
| Chunking strategy | `RecursiveCharacterTextSplitter` / `MarkdownHeaderTextSplitter` / `SemanticChunker` | **Recursive** (an toàn, đủ dùng trong 3h) |
| `CHUNK_SIZE` / `OVERLAP` | **Tự do** — test chỉ kiểm `size>0`, `overlap>0`, `overlap<size` | **800 / 100** |
| Embedding model | `all-MiniLM-L6-v2` / `BAAI/bge-m3` / `text-embedding-3-small` | **`BAAI/bge-m3`** (multilingual, tốt tiếng Việt) |
| Lexical method | BM25 (mặc định) / TF-IDF / Elasticsearch / Weaviate BM25 | **BM25Okapi** |
| Rerank method | Cross-encoder / MMR / **RRF** — *đề nói "chọn 1"* | **RRF k=60** (vì cần gộp 2 ranker, không tốn API) |
| `score_threshold` | Đề để mặc định `0.3` và ghi rõ *"calibrate bằng cách tự đo điểm cosine"* | **0.48** — TV2 đo lại ở CP3, được phép chỉnh |
| Eval framework | DeepEval / **RAGAS** / TruLens — *chọn 1* | **RAGAS** |
| LLM | OpenAI / Gemini / local / OpenRouter | **OpenRouter** |
| UI | Streamlit / Gradio / Chainlit | **Streamlit** (starter có sẵn) |

> ⚠️ **Sửa lại hiểu nhầm phổ biến:** con số `CHUNK_SIZE=800`, `OVERLAP=100`, `threshold=0.48`, `RRF` là **gợi ý trong bản hướng dẫn**, KHÔNG phải ràng buộc của barem. Test không kiểm các giá trị này. Nhóm được quyền chọn khác — **miễn là giải thích được vì sao khi coach hỏi**. Ta chốt các số trên để 5 người đồng bộ, không phải vì bị ép.

### 🎁 BONUS 20 điểm — chia sẵn cho từng người

| Tiêu chí bonus | Điểm | Giao cho |
|---|---|---|
| Implement HyDE hoặc Query Expansion | 5 | **TV2** (nằm sẵn trong Task 5) |
| Giải thích cơ chế lexical search khác BM25 trong demo | 5 | **TV3** (thêm TF-IDF để so sánh) |
| Deploy chatbot online (HF Spaces / Render) | 4 | **TV4** (nếu còn thời gian ở CP6) |
| Conversation memory (multi-turn) | 3 | **TV4** (*cũng là yêu cầu bắt buộc của chatbot*) |
| UI/UX chất lượng (hiển thị source, score, highlight) | 3 | **TV4** |

> Bonus 20đ là phần dễ ăn nhất: HyDE và TF-IDF chỉ tốn ~15 phút mỗi cái mà được 10đ.

### 📊 Barem thật (README) — khác với bản intro

| Thành phần | Tỷ trọng |
|---|---|
| Pipeline Task 1–10 (cá nhân, chấm bằng pytest) | **50%** |
| Bài nhóm (chatbot 18đ + evaluation 12đ) | **30%** |
| Bonus | **20%** |

> Bản intro ghi "50 cá nhân + 50 nhóm", README ghi "50 + 30 + 20". Chênh nhau đúng phần bonus — nghĩa là **muốn đủ 100 thì bắt buộc phải làm bonus**, đừng bỏ.
> Trọng số điểm từng task: Task 4 và 9 nặng nhất (7đ), Task 5/6/7 (6đ), Task 3 (4đ), Task 8 và 10 (4đ), Task 1/2 (3đ).
> README nói bài nhóm làm "1 trong 2 sản phẩm" (chatbot HOẶC eval), nhưng barem lại chấm điểm **cả hai**. → **Làm cả hai** cho an toàn.

---

## 3. HỢP ĐỒNG INTERFACE (chốt ở phút 0:10, không đổi sau đó)

Đây là thứ giúp 5 người code song song mà ghép vào vẫn chạy. Mọi hàm phải trả về **đúng** shape dưới đây (đã đối chiếu với `tests/test_individual.py`).

### Chuẩn "chunk dict" dùng chung toàn hệ thống

```
{
  "content":  str,          # nội dung đoạn văn
  "score":    float,        # điểm số (cosine / bm25 / rrf tuỳ tầng)
  "metadata": {
      "source":        str,   # tên file gốc, VD "returns-and-refunds-policy.md"
      "type":          str,   # "legal" | "news"
      "customer_role": str,   # "buyer" | "seller" | "both"   ← BẮT BUỘC (K4 variant)
      "chunk_id":      str
  }
}
```

### Chữ ký hàm bắt buộc

| Hàm | File | Chủ | Trả về |
|---|---|---|---|
| `semantic_search(query, top_k=10)` | task5 | TV2 | `list[chunk]`, **sort score giảm dần**, `score` là cosine gốc `[0,1]` |
| `lexical_search(query, top_k=10)` | task6 | TV3 | `list[chunk]`, sort giảm dần, score BM25 thô |
| `rerank_rrf(list_of_result_lists, top_k, k=60)` | task7 | TV3 | `list[chunk]` có `score` = RRF |
| `pageindex_search(query, top_k=5)` | task8 | TV3 | `list[chunk]`, mỗi item gắn `source="pageindex"` |
| `retrieve(query, top_k=5, score_threshold=0.48)` | task9 | TV1 | `list[dict]` có `content`, `score`, **`source` ∈ {"hybrid","pageindex"}** |
| `reorder_for_llm(chunks)` | task10 | TV4 | `list[chunk]` cùng độ dài, **phần tử đầu giữ nguyên** |
| `format_context(chunks)` | task10 | TV4 | `str` có chứa tên file nguồn |
| `generate_with_citation(query, top_k=5)` | task10 | TV4 | `dict` có key `answer` (str, khác rỗng), citation dạng `[Nguồn, Năm]`, thiếu evidence → `"I cannot verify this information"` |

**3 cái bẫy test hay rớt nhất:**

- `test_results_sorted_descending` → luôn `sorted(results, key=lambda r: r["score"], reverse=True)` trước khi return.
- `test_results_have_required_keys` (Task 9) → `source` phải là **đúng chuỗi** `"hybrid"` hoặc `"pageindex"`.
- `test_reorder_function_exists` → `reorder_for_llm` phải giữ `chunks[0]` ở vị trí `[0]` sau khi reorder (`front + back[::-1]`).

### Hằng số nhóm tự chốt (đề bài không ép — xem mục 2b — nhưng đã chốt thì không ai tự đổi)

```
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
EMBEDDING_MODEL = "BAAI/bge-m3"   (1024 chiều)
COLLECTION_NAME = "ecommerce_support_docs"
RRF_K           = 60
SCORE_THRESHOLD = 0.48    ← so với dense_results[0]["score"] (cosine GỐC), KHÔNG phải điểm RRF
```

> File starter đang để `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`, `SCORE_THRESHOLD=0.3`. TV2 và TV1 phải sửa lại theo bảng trên ở CP2.
> Nếu đổi `EMBEDDING_PROVIDER` trong `.env` → **cả nhóm phải xoá `chroma_db/` và index lại** (số chiều khác nhau, không tương thích).

---

## 4. Timeline theo checkpoint (180 phút)

### CP0 · 0:00–0:10 — Setup

| TV | Việc |
|---|---|
| TV1 | Tạo repo nhóm, tạo 5 branch `dev/tvX`, bật branch protection cho `main`, gửi `.env` (OPENROUTER_API_KEY) qua kênh riêng — **không commit `.env`** |
| TV2 | `python -m venv .venv` → `pip install -r requirements.txt` → test import `chromadb`, `sentence_transformers` |
| TV3 | Cài `markitdown[pdf]` + `playwright install chromium` (2 lỗi hay gặp nhất) |
| TV4 | `streamlit run app.py` chạy được (kể cả chưa có data) |
| TV5 | Test import `ragas`, `datasets` |

✅ **Pass CP0:** cả 5 máy import không lỗi. TV1 push commit đầu tiên chốt `.gitignore` + file này.

---

### CP1 · 0:10–0:35 — Thu thập & chuẩn hoá dữ liệu (Task 1–3)

| TV | Việc | Ghi vào |
|---|---|---|
| TV1 | Chốt interface ở mục 3, phân nguồn dữ liệu để không trùng, review PR | — |
| **TV2** | **Task 1** — tải ≥3 PDF chính sách (Đổi trả / Thanh toán / Quy định người bán) | `data/landing/legal/legal_01..03_*.pdf` |
| **TV2** | **Task 2** — crawl ≥5 bài help-center (TV3 gửi trước danh sách URL) | `data/landing/news/news_01..05_*.json` |
| **TV2** | **Task 3** — chạy `python -m src.task3_convert_markdown` | `data/standardized/legal|news/*.md` |
| TV3 | Gom danh sách URL help-center gửi TV2; dựng khung BM25 trong `src/task6_lexical_search.py`, test tạm bằng 2–3 file `.md` tự gõ tay ở local (không commit) | `dev/tv3` |
| TV5 | Đọc tài liệu, bắt đầu soạn 15 câu hỏi golden (dựa trên nội dung TV2 thu về) | nháp local |
| TV4 | Dựng khung UI `app.py`: sidebar, chat box, ô hiển thị nguồn (chưa nối pipeline) | `app.py` |

> **Chống conflict dữ liệu:** chỉ TV2 được ghi vào `data/`. Đặt tên file có prefix số như trên → không bao giờ trùng tên. TV3/TV5 cần thêm nguồn thì gửi URL cho TV2, không tự commit file vào `data/`.
> 🚨 **Repo hiện KHÔNG có sẵn dữ liệu** — cả 4 thư mục `data/` chỉ có `.gitkeep`. Không có bộ 11 file mẫu như đề bài nhắc tới. TV2 phải tự thu thập từ đầu, đây là đường găng (critical path) của cả nhóm.
> Trang help center dùng JavaScript render (SPA) → crawl về chỉ có tiêu đề, không có nội dung. Bị 403 hoặc SPA rỗng → đổi ngay sang nguồn khác (xem `SUGGESTED_TOPICS.md`, 9 chủ đề có sẵn), đừng ngồi debug quá 5 phút.
> **Phương án dự phòng nếu quá 0:25 vẫn chưa có dữ liệu:** copy thủ công nội dung 3 trang chính sách vào file `.md` rồi bỏ thẳng vào `data/standardized/legal/` — test Task 3 chỉ kiểm tra sự tồn tại của file `.md`, không kiểm tra cách tạo ra nó.

✅ **Pass CP1:** ≥3 file `legal/`, ≥5 file `news/`, có `.md` tương ứng. Test 1–3 (11 test) pass. Merge `feat/data` vào `main`.

---

### CP2 · 0:35–1:00 — Chunking, Indexing & Search cơ bản (Task 4–6)

| TV | Việc |
|---|---|
| TV1 | Xác nhận `CHUNK_SIZE=800`, `OVERLAP=100`, model `bge-m3`; review PR `feat/dense` + `feat/sparse` |
| **TV2** | **Task 4** — chunking + gắn `metadata["customer_role"]` + embed + index vào `chroma_db/` |
| **TV2** | **Task 5** — `semantic_search()` (cosine + tuỳ chọn HyDE) |
| **TV3** | **Task 6** — `lexical_search()` (BM25 / TF-IDF) |
| TV4 | Task 10 phần offline: viết `reorder_for_llm()` và `format_context()` (chưa cần LLM) |
| TV5 | Hoàn thiện `golden_dataset.json` 15 câu, chia đều buyer / seller / both / câu tổng hợp |

> ⚠️ `KeyError: 'customer_role'` là lỗi rớt test Task 4 phổ biến nhất — TV2 nhớ gán `metadata["customer_role"] = role` trong hàm chunking.
> TV3 làm BM25 đọc thẳng từ `data/standardized/*.md`, **không phụ thuộc `chroma_db/`** → TV3 không phải chờ TV2 index xong.

✅ **Pass CP2:** `chroma_db/` tồn tại; test Task 4, 5, 6 pass.

---

### CP3 · 1:00–1:20 — Reranking & Fallback (Task 7–8)

| TV | Việc |
|---|---|
| TV1 | Kiểm tra công thức RRF `1/(60+rank)`; chuẩn bị khung `retrieve()` |
| TV2 | Đo thực tế phân bố điểm cosine của 10 câu hỏi mẫu → báo TV1 để calibrate ngưỡng 0.48 |
| **TV3** | **Task 7** — `rerank_rrf()` gộp 2 danh sách rank |
| **TV3** | **Task 8** — `pageindex_search()` (điền `PAGEINDEX_API_KEY`; nếu không có key thì viết fallback trả về kết quả từ full-document search, vẫn gắn `source="pageindex"`) |
| TV4 | Nối `app.py` với `semantic_search` để test UI có dữ liệu thật |
| TV5 | Chuẩn bị 3 câu hỏi out-of-domain để kiểm tra fallback có kích hoạt không |

✅ **Pass CP3:** RRF gộp thành công; PageIndex trả kết quả. Test Task 7, 8 pass. Merge `feat/sparse`.

---

### CP4 · 1:20–1:45 — Pipeline hoàn chỉnh & Generation (Task 9–10) → **50 điểm cá nhân**

| TV | Việc |
|---|---|
| **TV1** | **Task 9** — `retrieve()`: semantic + BM25 → RRF; nếu `dense_results[0]["score"] < 0.48` → PageIndex fallback. Gắn `source` đúng chuẩn |
| **TV4** | **Task 10** — reorder `front + back[::-1]` + gọi LLM sinh câu trả lời có `[Nguồn: tên_file]` |
| TV2, TV3, TV5 | **Tự chạy full Task 1–10 trên branch `dev/tvX` của mình** để đạt 35/35 |
| TV1 | Đối soát: từng người screenshot `pytest tests/test_individual.py -v` → 35 passed |

> 🚨 **Bẫy chết người:** so ngưỡng 0.48 với **điểm cosine gốc** `dense_results[0]["score"]`, KHÔNG so với điểm RRF (RRF luôn ~0.016 nên fallback sẽ kích hoạt sai/không bao giờ kích hoạt).

✅ **Pass CP4:** cả 5 người đều `35 passed`. Merge `feat/pipeline` vào `main`.

---

### CP5 · 1:45–2:15 — Chatbot UI & Đánh giá RAGAS → **50 điểm nhóm**

| TV | Việc |
|---|---|
| TV1 | Chọn bản `src/` tốt nhất của nhóm làm bản chuẩn trên `main`; điều phối tiến độ |
| TV2 | Hỗ trợ TV4 nối `generate_with_citation()` vào `app.py`; rebuild `chroma_db/` bản cuối |
| **TV4** | Hoàn thiện `app.py`: chat UI, **conversation memory (follow-up) — BẮT BUỘC**, slider `top_k`, khu vực hiển thị nguồn, 5 câu hỏi gợi ý, badge cho biết đang dùng `hybrid` hay `pageindex` |
| TV3 | Chuẩn bị 1 **failure case có bằng chứng** (câu hỏi nào retrieval yếu, số liệu cosine bao nhiêu, fallback xử lý ra sao) |
| **TV5** | Chạy `python -m group_project.evaluation.eval_pipeline` → 4 chỉ số RAGAS; bảng A/B **Dense-only vs Hybrid+RRF**; viết `results.md` |

> Gặp `429 Too Many Requests` khi chạy RAGAS → tạm giảm golden_dataset xuống 5 câu để chạy thử, chạy full 15 câu 1 lần duy nhất ở cuối.

✅ **Pass CP5:** chatbot trả lời kèm danh sách nguồn; `results.md` có bảng A/B đầy đủ.

---

### CP6 · 2:15–3:00 — Demo & Nộp bài

| TV | Phần trình bày | Thời lượng |
|---|---|---|
| TV1 | Tổng quan kiến trúc RAG pipeline, giải thích lựa chọn thiết kế | ~2 phút |
| TV2 | Corpus, chunking 800/100, `customer_role`, embedding bge-m3 | ~1.5 phút |
| TV3 | Hybrid search + RRF + **failure case có bằng chứng** | ~2 phút |
| TV4 | Live demo Streamlit (chuẩn bị sẵn 3 câu: 1 dễ, 1 cần BM25 chính xác, 1 tổng hợp → kích fallback) | ~2 phút |
| TV5 | Kết quả RAGAS, so sánh Hybrid vs Dense-only, kết luận | ~2 phút |

✅ **Pass CP6:** demo xong, `git push origin main` với đầy đủ code + `results.md`.

---

## 5. Quy trình Git chuẩn cho mỗi thành viên

```bash
# Lần đầu
git clone <repo-nhom> && cd K4-Day08-3HQ
git checkout -b dev/tv3          # branch cá nhân, dùng suốt lab

# Khi làm phần bài NHÓM (file mình sở hữu)
git checkout main && git pull origin main
git checkout -b feat/sparse
# ... code chỉ trong file mình sở hữu ...
git add src/task6_lexical_search.py src/task7_reranking.py
git commit -m "[TV3][Task6-7] BM25 + RRF rerank"
git pull --rebase origin main    # BẮT BUỘC trước khi push
git push origin feat/sparse
# → mở PR, TV1 review & merge
```

**Nếu vẫn dính conflict** (thường do `.gitignore` sót hoặc ai đó phá luật ownership):

```bash
git status                       # xem file nào conflict
# Nếu là chroma_db/ hoặc __pycache__ → xoá luôn, không giữ:
git rm -r --cached chroma_db __pycache__
# Nếu là file code của người khác → giữ bản trên main:
git checkout --theirs <file>
git add <file> && git rebase --continue
```

**Thứ tự merge vào `main` (TV1 kiểm soát):**

```
feat/data (TV2) → feat/dense (TV2) → feat/sparse (TV3) → feat/pipeline (TV1) → feat/app (TV4) → feat/eval (TV5)
```

Merge theo đúng chiều phụ thuộc dữ liệu → không có PR nào phải chờ PR sau nó.

---

## 6. Checklist nộp bài

**Cá nhân (mỗi người, 50đ):**

- [ ] Branch `dev/tvX` có đủ Task 1–10
- [ ] `pytest tests/test_individual.py -v` → **35 passed**
- [ ] Screenshot kết quả test

**Nhóm (50đ):**

- [ ] `data/landing/legal/` ≥ 3 file · `data/landing/news/` ≥ 5 file
- [ ] `data/standardized/` có `.md` tương ứng, metadata `customer_role` đầy đủ
- [ ] `app.py` chạy được, trả lời có citation
- [ ] `group_project/evaluation/golden_dataset.json` ≥ 15 câu
- [ ] `group_project/evaluation/results.md` có bảng A/B 4 chỉ số RAGAS
- [ ] Chatbot có conversation memory (follow-up questions)
- [ ] Câu trả lời có citation dạng `[Nguồn, Năm]`; thiếu evidence → "I cannot verify this information"
- [ ] 1 failure case có bằng chứng số liệu
- [ ] `main` sạch: không có `.env`, `chroma_db/`, `__pycache__/`

**Bonus (20đ — đừng bỏ, cần để đủ 100):**

- [ ] TV2: HyDE hoặc Query Expansion trong Task 5 (5đ)
- [ ] TV3: thêm TF-IDF + giải thích khác biệt với BM25 trong demo (5đ)
- [ ] TV4: conversation memory (3đ) + UI hiển thị source/score/highlight (3đ)
- [ ] TV4: deploy HF Spaces nếu kịp (4đ)

---

## 7. Ma trận phụ thuộc — ai chờ ai

| Việc | Cần có trước | Người chờ |
|---|---|---|
| Task 4 (index) | Task 3 xong (`.md`) | TV2 chờ chính mình |
| Task 5 (semantic) | Task 4 xong | TV2 |
| Task 6 (BM25) | Task 3 xong (**không cần Task 4**) | TV3 — làm song song với TV2 |
| Task 7 (RRF) | chỉ cần *shape* của Task 5/6 → dùng dữ liệu giả để code trước | TV3 — **không chờ ai** |
| Task 9 (pipeline) | Task 5, 6, 7, 8 | TV1 — chờ lâu nhất, nên TV1 viết khung trước ở CP3 |
| Task 10 (generation) | Task 9 (chỉ cần chữ ký hàm) | TV4 — code `reorder`/`format_context` trước từ CP2 |
| `app.py` | Task 10 | TV4 |
| RAGAS eval | `app.py` + Task 9 | TV5 — soạn golden_dataset song song từ CP1 |

> Nhờ **hợp đồng interface chốt ở phút 0:10**, TV3/TV4/TV5 code trước bằng dữ liệu giả (mock) mà không cần đợi TV2 index xong. Đây là điểm quyết định để nhóm 5 người chạy song song trong 3 giờ.
