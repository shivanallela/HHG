# Model Harness Design — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1 — Implementation in Step 3)
> **Last Updated**: 2026-08-19

---

## Overview

The model harness is the structured execution framework that wraps the LLM inference step. It is NOT a simple `prompt → LLM → answer` call. It provides retries, timeouts, validation, grounding checks, tracing, and structured output.

## Harness Pipeline

```
┌──────────────────────────────────────────────────┐
│                 MODEL HARNESS                      │
│                                                    │
│  ┌────────────────────────────────────────────┐   │
│  │  1. Input Validation                        │   │
│  │  - Query present and non-empty              │   │
│  │  - Query within max length                  │   │
│  │  - Context provided                         │   │
│  │  - Trace ID assigned                        │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  2. Query Processing                        │   │
│  │  - Classify query type                      │   │
│  │  - Extract key entities                     │   │
│  │  - Determine expected answer format         │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  3. Context Validation                      │   │
│  │  - Verify context is non-empty              │   │
│  │  - Check relevance scores above threshold   │   │
│  │  - Trim context to token budget             │   │
│  │  - IF no relevant context → early exit      │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  4. Prompt Construction                     │   │
│  │  - System prompt with instructions          │   │
│  │  - Context passages with source markers     │   │
│  │  - User query                               │   │
│  │  - Output format instructions               │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  5. LLM Generation (with retries)           │   │
│  │  - Send prompt to configured LLM            │   │
│  │  - Timeout: 30s (configurable)              │   │
│  │  - Retry: up to 3 times with backoff        │   │
│  │  - On failure: return structured error       │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  6. Response Parsing                        │   │
│  │  - Extract answer text                      │   │
│  │  - Parse source citations                   │   │
│  │  - Extract confidence signal                │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  7. Grounding Check                         │   │
│  │  - Verify claims against retrieved context  │   │
│  │  - Flag unsupported statements              │   │
│  │  - Compute grounding score                  │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  8. Guardrail Check                         │   │
│  │  - Safety check on generated content        │   │
│  │  - Hallucination flag evaluation            │   │
│  │  - Confidence threshold enforcement         │   │
│  └──────────────────┬─────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │  9. Structured Output                       │   │
│  │  {                                          │   │
│  │    "answer": "...",                          │   │
│  │    "sources": [...],                         │   │
│  │    "confidence": 0.87,                       │   │
│  │    "grounding_score": 0.92,                  │   │
│  │    "trace_id": "uuid",                       │   │
│  │    "latency_ms": 142,                        │   │
│  │    "warnings": []                            │   │
│  │  }                                          │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
└──────────────────────────────────────────────────┘
```

## Retry & Timeout Strategy

```python
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0          # seconds
    max_delay: float = 10.0          # seconds
    backoff_factor: float = 2.0      # exponential
    timeout: float = 30.0            # per-attempt timeout

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)
```

### Retry Decision Matrix

| Error Type | Retry? | Notes |
|-----------|--------|-------|
| Network timeout | Yes | Transient, likely recovers |
| Rate limit (429) | Yes | With longer backoff |
| Server error (5xx) | Yes | Up to max_retries |
| Auth error (401/403) | No | Config issue, fail fast |
| Invalid response | Yes (once) | May be transient |
| Context too long | No | Reduce context, retry |

## Structured Input/Output

### Input Schema

```python
@dataclass
class HarnessInput:
    query: str                          # User's question
    context: list[RetrievedPassage]     # Retrieved passages
    trace_id: str                       # UUID for request tracking
    max_tokens: int = 512               # Max generation tokens
    temperature: float = 0.1            # Low for factual answers
    require_grounding: bool = True      # Enable grounding check
```

### Output Schema

```python
@dataclass
class HarnessOutput:
    answer: str                         # Generated answer
    sources: list[SourceCitation]       # Cited sources
    confidence: float                   # 0.0 – 1.0
    grounding_score: float              # 0.0 – 1.0
    trace_id: str                       # Same as input
    latency_ms: float                   # Total harness latency
    stage_latencies: dict[str, float]   # Per-stage breakdown
    warnings: list[str]                 # Any guardrail warnings
    status: str                         # "success" | "low_confidence" | "error"
```

## Failure Recovery

```
IF input validation fails:
    → Return error with specific field details

IF no relevant context:
    → Return: "I couldn't find enough relevant information..."
    → Set status = "insufficient_context"

IF LLM fails after all retries:
    → Return structured error with trace_id
    → Log full error chain

IF grounding score < threshold (0.5):
    → Return answer with warning flag
    → Set status = "low_confidence"

IF guardrails flag unsafe content:
    → Redact unsafe portions
    → Return with safety warning
```

## Logging & Tracing

Every harness invocation produces a trace log:

```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "stages": [
    {"name": "input_validation", "duration_ms": 1, "status": "pass"},
    {"name": "query_processing", "duration_ms": 3, "status": "pass"},
    {"name": "context_validation", "duration_ms": 2, "status": "pass"},
    {"name": "prompt_construction", "duration_ms": 1, "status": "pass"},
    {"name": "llm_generation", "duration_ms": 850, "status": "pass", "attempt": 1},
    {"name": "response_parsing", "duration_ms": 2, "status": "pass"},
    {"name": "grounding_check", "duration_ms": 15, "status": "pass", "score": 0.92},
    {"name": "guardrail_check", "duration_ms": 5, "status": "pass"}
  ],
  "total_latency_ms": 879,
  "status": "success"
}
```
