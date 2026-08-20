# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. **Case nào nên dùng multi-agent? Vì sao?**
   - **Các tác vụ nghiên cứu/tổng hợp phức tạp có tính decomposability thực sự:** Khi công việc đòi hỏi phải phân cấu trúc rõ ràng thành các nhiệm vụ con độc lập và chuyên biệt (ví dụ: Researcher thu thập tài liệu thô, Analyst đánh giá độc lập tính trung thực, Writer tổng hợp thành báo cáo). Việc tách biệt các vai trò giúp giảm xung đột ngữ cảnh trong prompt.
   - **Khi cần cơ chế tự sửa lỗi và kiểm tra chéo (Adversarial loops/Self-correction):** Việc có một Analyst hoặc Critic đánh giá độc lập dữ liệu từ Researcher giúp giảm đáng kể hiện tượng ảo giác (hallucination), kiểm chứng lại nguồn trích dẫn chéo, nâng điểm Quality score thực nghiệm từ 3.8 (baseline) lên 8.8 (multi-agent).
   - **Khi lượng tài liệu lớn vượt quá bộ nhớ làm việc hiệu quả của mô hình:** Các agent lấy dữ liệu song song và chỉ tóm tắt các notes cô đọng chuyển tiếp giúp vượt qua hiện tượng "lost-in-the-middle" của single agent.

2. **Case nào không nên dùng multi-agent? Vì sao?**
   - **Các tác vụ ngắn, hội thoại tuyến tính hoặc tóm tắt từ một tài liệu duy nhất:** Nơi một agent chuyên dụng có thể trả lời trực tiếp hoặc thông qua Chain-of-Thought ngắn mà không cần sự phối hợp phức tạp.
   - **Hạn chế nghiêm ngặt về ngân sách chi phí (Budget):** Luồng điều phối giữa các agent tiêu tốn lượng token gấp nhiều lần do phải truyền tải State nhiều lần qua lại (trong bài lab, chi phí tăng ~5.8 lần từ $0.000244 của baseline lên $0.001432).
   - **Yêu cầu phản hồi thời gian thực (Low Latency):** Hệ thống multi-agent phải thực hiện nhiều lời gọi API nối tiếp nhau thông qua quyết định của Supervisor làm tăng độ trễ hệ thống (từ 4.20 giây lên 14.10 giây).

