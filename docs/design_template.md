# Design Template

## Problem

Hệ thống cần xử lý câu hỏi nghiên cứu phức tạp đòi hỏi tìm kiếm thông tin, phân tích bằng chứng từ nhiều nguồn và tổng hợp câu trả lời có trích dẫn. Một agent đơn lẻ dễ bị loãng context khi phải làm tất cả: search → analyse → write trong một lần.

## Why multi-agent?

Single-agent gặp khó khăn khi:
- Context window bị chia sẻ cho nhiều nhiệm vụ khác nhau (search + analysis + writing)
- Không có independent verification — agent tự review output của mình
- Khó debug: không biết sai ở bước search, analysis hay viết

Multi-agent giải quyết bằng cách tách vai trò: Researcher chỉ tìm kiếm, Analyst chỉ đánh giá, Writer chỉ tổng hợp → mỗi agent có context chuyên biệt và rõ ràng.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing, guardrails, stop condition | ResearchState | Updated route_history | max_iterations exceeded |
| Researcher | Search offline corpus, tạo research notes | query, max_sources | sources, research_notes | No sources found, empty notes |
| Analyst | Phân tích evidence quality, detect conflicts | research_notes, sources | analysis_notes | False consensus, overgeneralization |
| Writer | Tổng hợp thành final report có citations | research_notes, analysis_notes | final_answer | Missing citations, hallucinated sources |

## Shared state

| Field | Lý do cần |
|---|---|
| `request` | Query gốc truyền xuyên suốt để agents không mất context |
| `iteration` | Guardrail — đếm số vòng để tránh vô hạn |
| `route_history` | Trace — biết flow thực tế |
| `sources` | Kết quả search — truyền cho Analyst và Writer cần biết nguồn |
| `research_notes` | Summary của Researcher — Analyst đọc và phân tích |
| `analysis_notes` | Assessment của Analyst — Writer dùng để viết có chiều sâu |
| `final_answer` | Output cuối — stop condition |
| `trace` | Debug — ghi lại mọi event |

## Routing policy

```
START → supervisor
supervisor:
  if iteration >= max_iterations → DONE
  if sources empty or no research_notes → researcher
  if no analysis_notes → analyst
  if no final_answer → writer
  else → DONE
researcher/analyst/writer → supervisor (loop)
supervisor DONE → END
```

## Guardrails

- Max iterations: 6 (cấu hình qua `MAX_ITERATIONS` trong .env)
- Timeout: 60s (cấu hình qua `TIMEOUT_SECONDS`)
- Retry: LLMClient tự retry 3 lần (tenacity exponential backoff)
- Fallback: Nếu LLM fail, state.errors ghi nhận và workflow dừng gracefully
- Validation: Pydantic schema cho mọi state field, sources phải là list[SourceDocument]

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| "When does multi-agent outperform single-agent?" | latency, cost, quality (0-10), citation coverage | Multi-agent: higher quality, higher cost |
| "What are failure modes in multi-agent systems?" | latency, cost, quality | Multi-agent nắm bắt nhiều failure modes hơn |

Baseline: single-agent LLM call thẳng, không có search/analysis step.
