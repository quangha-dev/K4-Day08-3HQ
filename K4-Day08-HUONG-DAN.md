# Hướng Dẫn Bài Lab Ngày 8 — K4 (E-commerce Support RAG)

---

## 1. Bài này là gì?

**K4-Day08-RAG-Pipeline-Starter** là bài lab Ngày 8 — bạn xây dựng một **RAG pipeline end-to-end** (từ thu thập dữ liệu đến chatbot trả lời có trích dẫn nguồn).

Chủ đề của K4: **Chính sách thương mại điện tử & hỗ trợ khách hàng** — chatbot trả lời câu hỏi về đổi trả/hoàn tiền, thanh toán, bảo mật, quy định người bán, theo dõi đơn hàng...

Dữ liệu mẫu trong repo được lấy từ **Shopee Vietnam Help Center** (`help.shopee.vn`).

Bài này **nối tiếp Ngày 7 (K4 Variant)** — cùng domain "E-commerce Policy / Customer Support", và **kế thừa yêu cầu metadata `customer_role`** từ Lab 07.

---

## 2. Cách làm bài: Nhóm tự chia task + Report cá nhân theo phần mình phụ trách

> **Có, cách này hoàn toàn ổn** — nhóm chia việc, mỗi bạn **own** vài task và **report cá nhân** phần đó. **Không bắt** ai phải code hết Task 1→10.

### Mô hình chung

| Ai | Làm gì | Nộp gì (cá nhân) |
|----|--------|------------------|
| **Cả nhóm** | 1 repo chung, pipeline Shopee/TMĐT chạy được | Demo + chatbot/eval |
| **Mỗi bạn** | Implement sâu task được giao | Report + pytest **task của mình** |
| **Không bắt buộc** | Mỗi người solo cả 10 task | — |

**Lưu ý K4:** Ai owner **Task 1 hoặc Task 4** phải đảm bảo metadata **`customer_role`** (`buyer`/`seller`/`both`) — ghi trong report cá nhân của người đó.

### Các bạn tự chia task — không có bảng role cố định

**Coach không gán sẵn ai làm task nào.** Nhóm **tự họp đầu buổi**, chia theo sở thích và cân bằng khối lượng:

1. Điền bảng phân công vào `group_project/README.md` (tên, MSSV, **task số**, file)
2. Chọn 1 bạn **điều phối** (nhắc tiến độ, merge code) — có thể trùng hoặc khác người code Task 9
3. Đảm bảo **Task 1 hoặc 4 có người nhận `customer_role`** — nhóm tự chỉ định, không cần “role Data” cố định

**Gợi ý khối công việc** (nhóm tự map vào thành viên):

| Khối | Task | Lưu ý K4 |
|------|------|----------|
| Data | 1, 2, 3 | Task 1: ghi `customer_role` từng tài liệu |
| Index & dense | 4, 5 | Task 4: gắn `customer_role` vào chunk |
| Sparse & fallback | 6, 7, 8 | BM25, RRF, PageIndex |
| Ghép & product | 9, 10, app, eval | Golden dataset nên có câu buyer + seller |

**Ví dụ chia (chỉ tham khảo — nhóm bạn có thể khác hoàn toàn):**
- 5 người: 3 + 2 + 2 + 3 + (app+eval) task... hoặc 2+2+2+2+2 — **tùy nhóm**.

Điều kiện duy nhất: **đủ 10 task có owner**, **không trùng**, **repo chung chạy được**.

### Report cá nhân — nộp gì?

1. **Task nhóm giao cho mình** (vd: Task 4, 5)
2. Implement gì (file, hàm, tham số)
3. **Vì sao** chọn cách đó
4. Screenshot pytest task của mình
5. Lỗi + cách fix
6. Cách phần mình nối pipeline

**Owner Task 1 hoặc 4:** thêm bảng **`customer_role`** từng tài liệu/chunk.

### Chấm điểm cá nhân (theo phần được giao)

| Tiêu chí | Trọng số gợi ý |
|----------|----------------|
| Task mình owner pass pytest | ~60% |
| Report cá nhân | ~30% |
| Demo — trả lời phần mình | ~10% |

Map điểm theo task owner, không bắt 35/35 nếu làm theo nhóm.

```bash
# Chạy test đúng task bạn owner
pytest tests/test_individual.py::TestTask4 -v
```

---

## 3. Điểm đặc biệt: `customer_role`

Đây là yêu cầu kế thừa từ Lab 07.

Mỗi tài liệu / chunk cần metadata:

| Giá trị | Ý nghĩa | Ví dụ |
|---------|---------|-------|
| `buyer` | Chỉ dành cho người mua | Chính sách trả hàng, thanh toán COD |
| `seller` | Chỉ dành cho người bán | Quy định đăng bán, phí sàn |
| `both` | Áp dụng cả hai | Chính sách bảo mật |

**Vì sao cần?**

Cùng một câu hỏi "phí" nhưng:
- Người **mua** quan tâm phí vận chuyển, phí COD
- Người **bán** quan tâm phí dịch vụ sàn, phí quảng cáo

Nếu không gắn nhãn, retriever có thể lấy nhầm chính sách của đối tượng khác → LLM trả lời sai.

**Làm ở đâu?**
- Task 1: Ghi chú role khi tải từng PDF
- Task 4: Gắn `metadata['customer_role'] = 'buyer'` (hoặc `seller` / `both`) vào mỗi chunk khi index ChromaDB

```python
# Ví dụ trong chunk_documents
chunks.append({
    "content": chunk_text,
    "metadata": {
        **doc["metadata"],
        "chunk_index": i,
        "customer_role": doc["metadata"].get("customer_role", "both"),
    },
})
```

**Lỗi hay gặp:** `KeyError: 'customer_role'` → bạn quên gán nhãn ở Task 4.

---

## 4. Toàn cảnh pipeline

Hãy tưởng tượng cả nhóm đang xây **chatbot hỗ trợ khách hàng Shopee**:

```
Thu thập chính sách → Chuẩn hóa → Chunk + gắn customer_role → Index ChromaDB
    → Semantic + BM25 → RRF Rerank → Fallback PageIndex
        → LLM trả lời có citation
```

### Task 1 — Tải văn bản chính sách TMĐT (≥ 3 file)

**Nhiệm vụ:** Tải **ít nhất 3** file PDF/DOCX về chính sách sàn TMĐT vào `data/landing/legal/`.

**Gợi ý nguồn (Shopee Help Center):**
- Chính sách trả hàng & hoàn tiền → thường `customer_role: buyer`
- Phương thức thanh toán → `buyer`
- Chính sách bảo mật → `both`
- Quy định đăng bán cho người bán → `seller`

**Đặt tên file rõ ràng:** `returns-refund-policy-shopee.pdf`, `payment-methods-shopee.pdf`...

**Pass khi:** ≥ 3 file trong `data/landing/legal/` → pytest `test_task1_*`

---

### Task 2 — Crawl bài hướng dẫn hỗ trợ (≥ 5 bài)

**Nhiệm vụ:** Crawl **ít nhất 5** bài hướng dẫn khách hàng vào `data/landing/news/`.

**Gợi ý chủ đề:**
- Theo dõi đơn hàng
- Đổi phương thức thanh toán
- Bằng chứng hoàn tiền
- Mua hàng xuyên biên giới

```bash
pip install crawl4ai
playwright install chromium
python src/task2_crawl_news.py
```

**Lưu ý K4:** Help center Shopee dùng JavaScript (SPA) — nếu crawl chỉ thấy tiêu đề, không có nội dung, hãy đổi URL bài khác cùng domain.

**Pass khi:** ≥ 5 file JSON trong `data/landing/news/` → pytest `test_task2_*`

---

### Task 3 — Convert sang Markdown

Dùng MarkItDown:

```bash
pip install "markitdown[pdf]"
python src/task3_convert_markdown.py
```

Output vào `data/standardized/legal/` và `data/standardized/news/`.

**Pass khi:** Có file `.md` tương ứng → pytest `test_task3_*`

---

### Task 4 — Chunking, Indexing + `customer_role`

**Nhiệm vụ:**
1. Chunk markdown (size/overlap bạn chọn, lab gợi ý 800/100)
2. **Gắn `customer_role` vào metadata mỗi chunk**
3. Embed `BAAI/bge-m3` → lưu ChromaDB collection `ecommerce_support_docs`

Collection name: `ecommerce_support_docs`. **Bắt buộc có `customer_role`** trong metadata chunk.

**Pass khi:** Vector store có data (kèm metadata đúng) → pytest `test_task4_*`

---

### Task 5 — Semantic Search

Viết `semantic_search(query, top_k)` — tìm theo **ngữ nghĩa**.

Ví dụ K4: câu "Gửi hàng lại như nào?" vẫn match chunk nói về "Quy trình Trả hàng/Hoàn tiền".

**Bonus (+5đ):** HyDE.

Test mẫu dùng query: `"ecommerce return policy"`.

**Pass khi:** Format + sorted → pytest `test_task5_*`

---

### Task 6 — Lexical Search (BM25)

Viết `lexical_search(query, top_k)` — tìm theo **từ khóa chính xác**.

Rất hiệu quả với: mã đơn hàng, mã voucher (`"Mã SPP123"`), tên chính sách cụ thể.

**Bonus (+5đ):** TF-IDF / Elasticsearch và giải thích trong demo.

**Pass khi:** Format đúng → pytest `test_task6_*`

---

### Task 7 — Reranking (RRF khuyến nghị)

Gộp kết quả Semantic + BM25 bằng RRF:

\[
RRF(d) = \sum \frac{1}{60 + rank(d)}
\]

**Vì sao không cộng trực tiếp cosine + BM25?** Hai thang điểm khác nhau hoàn toàn. RRF chỉ dùng **thứ hạng**, công bằng hơn.

**Pass khi:** Rerank hoạt động → pytest `test_task7_*`

---

### Task 8 — PageIndex Fallback

Tích hợp PageIndex cho câu hỏi **tổng hợp**, ví dụ:

> "Tóm tắt toàn bộ quy trình khiếu nại trả hàng cho Người bán?"

Chunk nhỏ có thể mất context — PageIndex đọc theo cấu trúc mục lục.

**Pass khi:** `pageindex_search` trả kết quả → pytest `test_task8_*`

---

### Task 9 — Retrieval Pipeline

Viết `retrieve()` nối toàn bộ luồng + fallback:

```
Semantic + BM25 → RRF → Rerank
Nếu dense_results[0]["score"] < threshold → PageIndex
```

**BẪY:** Không so threshold với điểm RRF (~0.016). Phải so **cosine score gốc**.

Test mẫu: `"ecommerce return policy"`.

**Pass khi:** Pipeline + fallback → pytest `test_task9_*`

---

### Task 10 — Generation có Citation

1. Reorder chunks: `front + back[::-1]` (chống lost-in-the-middle)
2. Prompt bắt LLM trích dẫn `[Shopee Help Center, 2026]` hoặc tương tự
3. Không đủ evidence → `"I cannot verify this information"`

Test mẫu: `"What payment methods does Shopee support?"`

**Pass khi:** Citation + reorder → pytest `test_task10_*`

---

## 5. Chấm điểm tổng thể

| Thành phần | Điểm | Ghi chú |
|-----------|------|---------|
| **Cá nhân (theo role)** | **50%** | Pytest task được giao + report |
| **Bài nhóm** | **30%** | Chatbot 🛒 + eval |
| **Bonus** | **20%** | HyDE, filter `customer_role` trên UI... |

- **Solo:** 35/35 test ≈ full 50đ cá nhân.  
- **Nhóm:** pass test **task của role** + report → map vào 50đ.

---

## 6. Bài nhóm

### Option A: Chatbot Streamlit (`app.py`)

Starter có sẵn giao diện 🛒 **E-commerce Support RAG**.

**Câu hỏi gợi ý trong sidebar:**
- "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?"
- "Shopee hỗ trợ những phương thức thanh toán nào?"
- "Quy định về đăng bán sản phẩm cho người bán?"
- "Cách mua hàng trên Shopee của quốc gia khác?"

**Nâng cao (bonus):** Thêm dropdown chọn `customer_role` (buyer/seller) để filter retrieval — tận dụng metadata từ Lab 07.

### Option B: RAG Evaluation

**Golden dataset K4 — ví dụ câu hỏi:**
- "Người mua có thể yêu cầu trả hàng trong bao lâu sau khi nhận hàng?"
- "Shopee hỗ trợ những phương thức thanh toán nào?"
- "Người bán không được đăng bán những sản phẩm nào?"

Cần ≥ 15 cặp Q&A, chạy RAGAS/DeepEval, so sánh A/B, viết `results.md`.

---

## 7. Lộ trình thời gian

| Giai đoạn | Việc | Ghi chú K4 |
|-----------|------|------------|
| 0:00–0:15 | Họp **tự chia task**, ghi README | Chốt ai lo `customer_role` (Task 1/4) |
| 0:15–0:45 | Task 1–3 | Data owner |
| 0:45–1:30 | Task 4–6 | Task 4: gắn `customer_role` trước khi ghép Task 9 |
| 1:30–2:00 | Task 7–8 | |
| 2:00–2:20 | Task 9–10 | |
| 2:20+ | App, eval, report cá nhân, demo | |

Thứ tự pipeline là cố định; **ai làm giai đoạn nào do nhóm quyết**.

---

## 8. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `KeyError: 'customer_role'` | Quên gán nhãn ở Task 4 | Thêm vào metadata khi chunk/index |
| Crawl Shopee chỉ có tiêu đề | SPA render JS | Đổi URL bài khác hoặc dùng data mẫu repo |
| Trả lời lẫn buyer/seller | Không filter theo role | Filter ChromaDB theo `customer_role` |
| Fallback không chạy | So nhầm RRF vs cosine | Dùng `dense_results[0]["score"]` |
| `MissingDependencyException` PDF | Thiếu markitdown pdf extra | `pip install "markitdown[pdf]"` |

---

## 9. Checklist trước khi nộp

**Cá nhân:** đã thống nhất task với nhóm → pytest task mình → report (owner Task 1/4: có `customer_role`) → demo phần mình.

**Nhóm:** README có phân công → repo chung pass full pytest → chatbot + eval.

---

**Tóm lại K4:** **Tự chia task** trong nhóm; owner Task 1/4 lo `customer_role`; report theo phần mình — không cần ôm 10 task.

```bash
pytest tests/test_individual.py::TestTask4 -v   # nếu bạn owner Task 4
```
