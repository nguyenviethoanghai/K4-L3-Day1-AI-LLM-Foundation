# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, phản hồi chuyển từ tính nhất quán, khách quan sang ngẫu nhiên và sáng tạo hơn. 
Ở mức 0.0 và 0.5, mô hình thường trả lời các sự thật phổ biến (như xuất khẩu cà phê, hang Sơn Đoòng) với câu từ ổn định, ít thay đổi giữa các lần chạy. 
Ở mức 1.0 đến 1.5, cách hành văn bay bổng và lựa chọn từ ngữ đa dạng hơn
Ở 1.5 bắt đầu có dấu hiệu dài dòng và tiềm ẩn rủi ro hallucination .

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.1. Chatbot hỗ trợ khách hàng yêu cầu tính chính xác, nhất quán và tin cậy cao nhất khi cung cấp thông tin sản phẩm, chính sách bảo hành hay giá cả. Thiết lập temperature thấp triệt tiêu tính ngẫu nhiên, giúp giảm thiểu tối đa hiện tượng hallucination và đảm bảo câu trả lời luôn đồng nhất giữa các khách hàng khác nhau.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng output token mỗi ngày là 10.000 × 3 × 350 = 10.500.000 tokens. Theo bảng giá, GPT-4o ($0.010/1K) tốn $105/ngày trong khi GPT-4o-mini ($0.0006/1K) chỉ tốn $6.3/ngày; do đó GPT-4o đắt hơn khoảng 16.7 lần . Trường hợp xứng đáng dùng GPT-4o: trợ lý phân tích hợp đồng pháp lý, tư vấn y tế hoặc giải quyết các bài toán logic lập trình phức tạp đòi hỏi độ chính xác tuyệt đối. Trường hợp nên dùng mini: chatbot hỗ trợ giải đáp FAQ cơ bản, phân loại ý định người dùng (intent routing) hoặc tóm tắt hội thoại ngắn quy mô lớn.


## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với vai giáo viên tiểu học, phản hồi ngắn gọn, dùng từ ngữ thân thuộc và ví blockchain như một cuốn sổ ghi chép chung của cả lớp mà ai cũng có một bản copy nên không ai gian lận được. Ngược lại, vai chuyên gia tài chính dùng phản hồi dài, cấu trúc trang trọng và sử dụng thuật ngữ chuyên môn như sổ cái phân tán (distributed ledger), mã hóa bất đối xứng, cơ chế đồng thuận (consensus) và tính bất biến (immutability). System prompt hoạt động như một bộ định hướng ngữ cảnh (context steering), điều chỉnh toàn diện từ vựng, tông giọng, độ sâu tri thức và phương pháp diễn giải của mô hình.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, ước lượng `số từ / 0.75 ≈ 133 token`, trong khi `count_tokens` (tiktoken) đo được thực tế khoảng 165 – 190 token, tức là chênh lệch cao hơn ước lượng khoảng 25% – 40%. Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì các thuật toán phân tách token (như BPE) được tối ưu hóa theo tần suất kho ngữ liệu tiếng Anh .Trong khi đó tiếng Việt có các ký tự đặc thù và nguyên âm có dấu (ă, â, đ, ê, ô, ơ, ư cùng dấu thanh), buộc tokenizer phải bẻ nhỏ một âm tiết thành 2 đến 3 subwords hoặc byte tokens riêng biệt.


## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng đối thoại trực tiếp tương tác với người dùng (chatbot hỗ trợ, trợ lý CLI, soạn thảo văn bản tương tác), nơi chỉ số Time-to-First-Token (TTFT) quyết định trực tiếp cảm giác mượt mà và giảm cảm giác sốt ruột khi người dùng có thể đọc câu trả lời ngay sau 0.5s thay vì chờ trọn vẹn 5 - 10s. 
Ngược lại, non-streaming phù hợp hơn trong các tác vụ xử lý chạy nền , các endpoint API trả về JSON/cấu trúc dữ liệu cố định (như Function Calling/Tool Use) đòi hỏi toàn bộ payload phải hoàn chỉnh mới parse được, hoặc các hệ thống cần kiểm duyệt nội dung an toàn toàn bộ phản hồi trước khi gửi về client.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có lợi thế là kéo giãn khoảng cách giữa các lần thử lại theo cấp số nhân (0.1s -> 0.2s -> 0.4s...), giúp mật độ request giảm dần theo thời gian và tạo cơ hội cho server đang nghẽn giải phóng tài nguyên. Nếu hàng nghìn client cùng retry với thời gian cố định (ví dụ đúng 1 giây), toàn bộ các client sẽ dội request ngược lại server vào cùng một thời điểm sau mỗi giây, gây ra hiệu ứng Retry Storm. Hậu quả là server bị đánh sập liên tục và không bao giờ phục hồi được trạng thái hoạt động bình thường.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona được chọn: "Trợ giảng AI thân thiện, chuyên hỗ trợ sinh viên học môn Lập trình AI". System prompt: *"Bạn là một trợ giảng AI thân thiện của khóa học AI, luôn giải thích khái niệm rõ ràng, có ví dụ minh họa và trả lời ngắn gọn bằng tiếng Việt chuẩn mực."* Cụm từ "trả lời ngắn gọn" được đưa vào để hạn chế phản hồi lan man trên giao diện dòng lệnh CLI, đồng thời tiết kiệm đáng kể lượng output token tích lũy. Cụm từ "bằng tiếng Việt chuẩn mực" giúp cố định ngôn ngữ phản hồi, ngăn chặn tình trạng mô hình tự động chuyển sang tiếng Anh khi người dùng hỏi các thuật ngữ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là cửa sổ ngữ cảnh ngắn (chỉ giữ 3 lượt gần nhất, cắt mất thông tin cũ) và không có bộ nhớ dài hạn giữa các phiên làm việc độc lập. Đề xuất cải thiện: Triển khai cơ chế "Tóm tắt ngữ cảnh tự động" (Context Summarization) kết hợp lưu trữ file: Khi danh sách history vượt quá 6 tin nhắn, ta dùng mô hình nhỏ (như `gpt-4o-mini`) tóm tắt các tin nhắn cũ thành một đoạn tóm lược súc tích và lưu vào biến `summary` nằm trong system prompt. Đồng thời, lưu trữ thông tin cá nhân và lịch sử phiên vào file JSON hoặc SQLite trên máy tính để khi người dùng mở lại CLI ở lần tiếp theo, trợ lý vẫn nhận diện được ngữ cảnh cũ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
