# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

---

# BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (BENCHMARK ANALYSIS REPORT)

Dưới đây là kết quả đo lường và phân tích thực tế khi chạy thử nghiệm Baseline Agent và Advanced Agent ở chế độ **LIVE** kết nối trực tiếp với mô hình **Gemini 2.5 Flash** thật.

### 1. Kết Quả Đo Lường Live Mode

#### 1.1. Standard Benchmark (`conversations.json` - 10 phiên hội thoại ngắn)
| Agent Name | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent (Std)** | 1887 | 14494 | **0.11** | 0.11 | 0 | 0 |
| **Advanced Agent (Std)** | 19901 | 59960 | **0.82** | 0.82 | 271 | 30 |

#### 1.2. Long-Context Stress Benchmark (`advanced_long_context.json` - stress test hội thoại siêu dài)
| Agent Name | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent (Stress)** | 2346 | 36484 | **0.00** | 0.00 | 0 | 0 |
| **Advanced Agent (Stress)** | 6106 | **17048** | **1.00** | 1.00 | 180 | **26** |

---

### 2. Phân Tích & Lý Giải Kết Quả (Reflection & Trade-offs)

#### 2.1. Tại sao Advanced Agent có khả năng Recall vượt trội?
* **Baseline Agent**: Khi chuyển sang thread mới để hỏi lại thông tin cũ, điểm Recall gần như bằng 0 (chỉ đạt 0.11 ở Standard test do sự ngẫu nhiên/trùng lặp). Điều này chứng minh Baseline Agent chỉ nhớ trong phạm vi phiên hiện tại và hoàn toàn quên chéo phiên.
* **Advanced Agent**: Đạt điểm Recall xuất sắc (**0.82** ở Standard và **1.00** ở Stress). Lớp bộ nhớ **`User.md` bền vững** cho phép Agent liên tục trích xuất thông tin người dùng từ tin nhắn và lưu trữ lâu dài. Khi chuyển sang thread mới, hệ thống tự động tải nội dung `User.md` vào ngữ cảnh prompt, giúp Agent trả lời chính xác các câu hỏi cá nhân hóa và nhận biết chính xác thông tin đính chính (ví dụ: đổi nơi ở từ Huế sang Đà Nẵng).

#### 2.2. Đánh đổi (Trade-off) về chi phí Prompt Token ở hội thoại ngắn
* Tại cuộc hội thoại ngắn (Standard Benchmark), Advanced Agent tiêu thụ nhiều Prompt Token hơn Baseline Agent (59,960 tokens so với 14,494 tokens).
* **Lý do**: Advanced Agent luôn mang theo hồ sơ người dùng `User.md` và tóm tắt hội thoại cũ trong System Prompt ở mỗi lượt chat. Khi hội thoại chưa đủ dài để tạo ra sự chênh lệch lớn, lượng thông tin mang thêm này làm gia tăng đáng kể số lượng Prompt Token processed.

#### 2.3. Lợi thế tối ưu của Compact Memory ở hội thoại dài
* Ở cuộc hội thoại stress test siêu dài, Baseline Agent tích lũy toàn bộ tin nhắn thô qua mỗi lượt, khiến Prompt Token tăng vọt lên tới 36,484 tokens.
* Advanced Agent kích hoạt nén bộ nhớ 26 lần (`Compactions = 26`) giúp nén các tin nhắn cũ vượt ngưỡng thành một đoạn tóm tắt (`summary`). Nhờ đó, lượng Prompt Token processed chỉ còn **17,048 tokens** (tiết kiệm hơn **53% chi phí** so với Baseline) mà vẫn giữ được chất lượng phản hồi tối đa (Recall & Quality đạt 1.00).

#### 2.4. Sự tăng trưởng của bộ nhớ (Memory Growth) & Hướng khắc phục
* Kích thước file hồ sơ người dùng `User.md` tăng thêm khoảng 180 - 271 bytes sau quá trình chạy. Ở quy mô thực tế, file này sẽ phình to liên tục.
* **Giải pháp nâng cao**: Để tránh tràn bộ nhớ trong thực tế, cần áp dụng cơ chế **Memory Decay** (giảm độ ưu tiên hoặc xóa bỏ thông tin cũ) hoặc **Confidence Threshold** (chỉ lưu các facts có độ tự tin cao) và giải quyết xung đột thông tin (Conflict/Correction Handling) để tối ưu dung lượng đĩa và token.

