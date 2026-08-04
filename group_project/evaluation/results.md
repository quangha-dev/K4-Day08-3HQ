# RAGAS Evaluation Results

- Evaluated rows: 30
- Pipeline errors: 0
- Judge model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`

## Aggregate Matrix

| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |
|---|---:|---:|---:|---:|---:|
| hybrid | 0.6556 | 0.3995 | 0.7333 | 0.7803 | 0.6422 |
| semantic | 0.7000 | 0.4285 | 0.8000 | 0.7956 | 0.6810 |

## Per-question Matrix

| Config | ID | Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---:|---|---:|---:|---:|---:|
| hybrid | 1 | Shopee Việt Nam đang hỗ trợ những hình thức thanh toán nào cho người mua? | 1.0000 | 0.8070 | 1.0000 | 1.0000 |
| hybrid | 2 | Phương thức thanh toán bằng Thẻ Tín dụng hoặc Thẻ Ghi nợ áp dụng cho đơn hàng có giá trị tối thiểu là bao nhiêu? | 1.0000 | 0.6658 | 1.0000 | 1.0000 |
| hybrid | 3 | Hạn mức thanh toán cho đơn hàng trên Shopee qua phương thức Apple Pay nằm trong khoảng nào? | 0.5000 | 0.7840 | 1.0000 | 1.0000 |
| hybrid | 4 | Hạn mức giá trị đơn hàng tối đa đối với phương thức thanh toán Google Pay trên Shopee là bao nhiêu? | 1.0000 | 0.8020 | 1.0000 | 1.0000 |
| hybrid | 5 | Phương thức SPayLater hỗ trợ các loại kỳ hạn thanh toán trả sau nào? | 1.0000 | 0.5503 | 1.0000 | 0.9500 |
| hybrid | 6 | Chính sách Trả hàng và Hoàn tiền Shopee áp dụng đối với những đối tượng nào? | 0.5000 | 0.5519 | 1.0000 | 1.0000 |
| hybrid | 7 | Người mua chỉ có thể gửi yêu cầu trả hàng/hoàn tiền trong những trường hợp nào? | 0.3333 | 0.3794 | 1.0000 | 1.0000 |
| hybrid | 8 | Yêu cầu hoàn tiền khi gặp sự cố giao dịch Ví ShopeePay tại cửa hàng sẽ được xử lý trong bao lâu? | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| hybrid | 9 | Khi phát hiện giao dịch lạ phát sinh từ Shopee trên thẻ ngân hàng, người mua cần kiểm tra mục nào đầu tiên? | 1.0000 | 0.6741 | 1.0000 | 1.0000 |
| hybrid | 10 | Nếu tài khoản Shopee có dấu hiệu bị lạm dụng hoặc truy cập trái phép, người dùng cần làm gì ngay lập tức? | 1.0000 | 0.0000 | 1.0000 | 0.9500 |
| hybrid | 11 | Shopee Việt Nam chỉ hỗ trợ đặt và giao hàng cho những người mua có địa chỉ ở đâu? | 1.0000 | 0.7778 | 1.0000 | 0.8042 |
| hybrid | 12 | Người mua có mấy cách chính để tiến hành đặt hàng trên ứng dụng Shopee sau khi chọn được sản phẩm? | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| hybrid | 13 | Thời tiết hôm nay ở Hà Nội như thế nào và chiều nay có mưa không? | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | 14 | Hướng dẫn cách nấu món Phở bò gia truyền Hà Nội gồm những nguyên liệu gì? | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | 15 | Làm thế nào để đăng ký tài khoản làm tài xế giao hàng trên ứng dụng Grab? | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| semantic | 1 | Shopee Việt Nam đang hỗ trợ những hình thức thanh toán nào cho người mua? | 1.0000 | 0.8070 | 1.0000 | 1.0000 |
| semantic | 2 | Phương thức thanh toán bằng Thẻ Tín dụng hoặc Thẻ Ghi nợ áp dụng cho đơn hàng có giá trị tối thiểu là bao nhiêu? | 1.0000 | 0.5266 | 1.0000 | 0.6792 |
| semantic | 3 | Hạn mức thanh toán cho đơn hàng trên Shopee qua phương thức Apple Pay nằm trong khoảng nào? | 1.0000 | 0.7840 | 1.0000 | 1.0000 |
| semantic | 4 | Hạn mức giá trị đơn hàng tối đa đối với phương thức thanh toán Google Pay trên Shopee là bao nhiêu? | 1.0000 | 0.7914 | 1.0000 | 0.8042 |
| semantic | 5 | Phương thức SPayLater hỗ trợ các loại kỳ hạn thanh toán trả sau nào? | 1.0000 | 0.5501 | 1.0000 | 1.0000 |
| semantic | 6 | Chính sách Trả hàng và Hoàn tiền Shopee áp dụng đối với những đối tượng nào? | 0.5000 | 0.5550 | 1.0000 | 1.0000 |
| semantic | 7 | Người mua chỉ có thể gửi yêu cầu trả hàng/hoàn tiền trong những trường hợp nào? | 0.3333 | 0.3803 | 1.0000 | 1.0000 |
| semantic | 8 | Yêu cầu hoàn tiền khi gặp sự cố giao dịch Ví ShopeePay tại cửa hàng sẽ được xử lý trong bao lâu? | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| semantic | 9 | Khi phát hiện giao dịch lạ phát sinh từ Shopee trên thẻ ngân hàng, người mua cần kiểm tra mục nào đầu tiên? | 1.0000 | 0.6741 | 1.0000 | 1.0000 |
| semantic | 10 | Nếu tài khoản Shopee có dấu hiệu bị lạm dụng hoặc truy cập trái phép, người dùng cần làm gì ngay lập tức? | 1.0000 | 0.5810 | 1.0000 | 1.0000 |
| semantic | 11 | Shopee Việt Nam chỉ hỗ trợ đặt và giao hàng cho những người mua có địa chỉ ở đâu? | 1.0000 | 0.7778 | 1.0000 | 1.0000 |
| semantic | 12 | Người mua có mấy cách chính để tiến hành đặt hàng trên ứng dụng Shopee sau khi chọn được sản phẩm? | 0.6667 | 0.0000 | 1.0000 | 1.0000 |
| semantic | 13 | Thời tiết hôm nay ở Hà Nội như thế nào và chiều nay có mưa không? | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| semantic | 14 | Hướng dẫn cách nấu món Phở bò gia truyền Hà Nội gồm những nguyên liệu gì? | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| semantic | 15 | Làm thế nào để đăng ký tài khoản làm tài xế giao hàng trên ứng dụng Grab? | 0.0000 | 0.0000 | 0.0000 | 0.4500 |

## Nhận xét: Vì sao Hybrid thấp hơn Semantic?

Trong lần đánh giá này, Hybrid đạt **0.6422**, thấp hơn Semantic (**0.6810**) **0.0388 điểm**. Nguyên nhân không phải BM25 luôn kém, mà do cách fusion hiện tại chưa phù hợp với corpus và bộ câu hỏi.

1. **Golden dataset thiên về factoid có section rõ ràng.** Các câu hỏi Apple Pay, Google Pay, SPayLater, thẻ tín dụng và hoàn tiền gần với heading/nội dung gốc; embedding `text-embedding-3-small` đã lấy đúng section nên BM25 bổ sung rất ít recall.
2. **BM25 đưa chunk nhiễu vào top-5.** Lexical search ưu tiên từ trùng khớp nhưng không hiểu đúng ý định. Sau RRF, một số chunk sparse từ tài liệu khác thay thế chunk dense liên quan, làm Context Precision giảm.
3. **Hai retriever đang dùng cách chunk khác nhau.** Dense dùng chunk 500 ký tự từ Task 4, còn BM25 dùng chunk heading tối đa 1.000 ký tự. Vì vậy cùng bằng chứng thường không có cùng identity; nhiều kết quả chỉ có `dense` hoặc `sparse` trong `raw_scores` thay vì được cả hai ranker cùng xác nhận.
4. **Điểm RRF quá sát nhau.** Với `k=60`, rank 1 chỉ đạt khoảng 0.01639 và rank 5 khoảng 0.01538. Một chunk BM25 rank cao nhưng ít liên quan có thể đứng trên chunk dense chứa đáp án. Hiện chưa có cross-encoder/query-aware reranker sau fusion để loại nhiễu này.
5. **Các câu ngoài phạm vi làm Hybrid nhạy với keyword hơn.** Câu thời tiết, nấu phở và Grab vẫn kích hoạt BM25 bởi các từ phổ biến; các sparse chunk không liên quan làm giảm Faithfulness/Precision.

### Bằng chứng nổi bật

- Câu 8 (hoàn tiền ShopeePay trong bao lâu): Context Recall của Hybrid là **0.0000**, trong khi Semantic là **1.0000**.
- Câu 10 (tài khoản bị truy cập trái phép): Answer Relevancy của Hybrid là **0.0000**, Semantic là **0.5810**.
- Câu 11 (địa chỉ giao hàng): Context Precision của Hybrid là **0.8042**, Semantic là **1.0000**.
- Trung bình Context Recall giảm từ **0.8000** xuống **0.7333**, cho thấy fusion đã làm mất một số evidence tốt.

### Hướng cải thiện

- Cho BM25 dùng chính xác corpus/chunk ID của Task 4 để dense và sparse fusion trên cùng đơn vị tài liệu.
- Tăng candidate pool trước fusion, nhưng chỉ lấy top-5 sau một cross-encoder hoặc LLM reranker.
- Thêm trọng số cho RRF hoặc weighted RRF để ưu tiên dense trên các câu hỏi diễn đạt tự nhiên.
