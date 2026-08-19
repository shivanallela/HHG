# System Architecture — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1)
> **Last Updated**: 2026-08-19

---

## Overview

This document describes the end-to-end architecture of the Voice-Enabled RAG system for the HH Goa 2026 Shortlisting Task 2.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                                │
│                    (React + Vite + TailwindCSS)                        │
│                                                                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────────┐    │
│  │  Voice In   │    │  Text In   │    │       Answer Display       │    │
│  └─────┬──────┘    └─────┬──────┘    └────────────────────────────┘    │
└────────┼─────────────────┼────────────────────────────────────────────┘
         │                 │
         ▼                 │
┌─────────────────┐        │
│  Speech-to-Text │        │       ┌─────────────────────────────────┐
│  (Sarvam /      │        │       │         FLASK BACKEND           │
│   ElevenLabs)   │        │       │                                 │
└────────┬────────┘        │       │  ┌───────────────────────────┐  │
         │                 │       │  │    Request Middleware       │  │
         ▼                 ▼       │  │  - Request ID generation    │  │
┌────────────────────────────┐     │  │  - Logging & latency start  │  │
│     Query Processing       │◄────┤  │  - Input sanitization       │  │
│  - Normalize text          │     │  └───────────┬───────────────┘  │
│  - Language detection      │     │              │                   │
│  - Query classification    │     │              ▼                   │
└────────┬───────────────────┘     │  ┌───────────────────────────┐  │
         │                         │  │    Input Validation        │  │
         ▼                         │  │  - Unsafe content check     │  │
┌────────────────────────────┐     │  │  - Off-topic detection      │  │
│     Retrieval Pipeline     │     │  │  - Query length validation   │  │
│                            │     │  └───────────┬───────────────┘  │
│  ┌──────────────────────┐  │     │              │                   │
│  │  Query Embedding     │  │     │              ▼                   │
│  │  (Sentence-BERT /    │  │     │  ┌───────────────────────────┐  │
│  │   HF Embedding)      │  │     │  │    Retrieval               │  │
│  └──────────┬───────────┘  │     │  │  - Embed query              │  │
│             │              │     │  │  - Vector search (FAISS)    │  │
│             ▼              │     │  │  - Top-K candidates         │  │
│  ┌──────────────────────┐  │     │  │  - Metadata filtering       │  │
│  │  Vector Search       │  │     │  │  - Optional reranking       │  │
│  │  (FAISS Index)       │  │     │  └───────────┬───────────────┘  │
│  └──────────┬───────────┘  │     │              │                   │
│             │              │     │              ▼                   │
│             ▼              │     │  ┌───────────────────────────┐  │
│  ┌──────────────────────┐  │     │  │    Context Validation      │  │
│  │  Optional Reranking  │  │     │  │  - Relevance threshold      │  │
│  │  (Cross-encoder)     │  │     │  │  - Context deduplication    │  │
│  └──────────┬───────────┘  │     │  │  - Token budget management  │  │
│             │              │     │  └───────────┬───────────────┘  │
└─────────────┼──────────────┘     │              │                   │
              │                    │              ▼                   │
              ▼                    │  ┌───────────────────────────┐  │
┌────────────────────────────┐     │  │    LLM Generation          │  │
│     RAG Generation         │     │  │  - Prompt construction      │  │
│  - Context assembly        │     │  │  - Model inference          │  │
│  - Prompt construction     │     │  │  - Retries & timeout        │  │
│  - LLM inference           │     │  │  - Fallback handling        │  │
│  - Response parsing        │     │  └───────────┬───────────────┘  │
└────────┬───────────────────┘     │              │                   │
         │                         │              ▼                   │
         ▼                         │  ┌───────────────────────────┐  │
┌────────────────────────────┐     │  │    Post-Processing         │  │
│     Guardrails             │     │  │  - Grounding verification   │  │
│  - Grounding check         │     │  │  - Guardrail checks         │  │
│  - Hallucination detection │     │  │  - Response formatting      │  │
│  - Safety filtering        │     │  │  - Latency measurement      │  │
│  - Confidence scoring      │     │  └───────────┬───────────────┘  │
└────────┬───────────────────┘     │              │                   │
         │                         └──────────────┼───────────────────┘
         ▼                                        │
┌────────────────────────────┐                    ▼
│     Final Answer           │         ┌───────────────────┐
│  - Structured JSON         │         │    Response        │
│  - Confidence score        │         │  - Answer text      │
│  - Source citations        │         │  - Sources           │
│  - Latency metadata        │         │  - Confidence        │
└────────────────────────────┘         │  - Latency stats     │
                                       └───────────────────┘
```

## Error Handling Strategy

| Stage | Error Type | Handling |
|-------|-----------|----------|
| STT | Transcription failure | Return error with retry suggestion |
| Query Processing | Empty/invalid query | Return validation error |
| Embedding | Model failure | Retry up to 3 times, then fallback |
| Vector Search | Index unavailable | Return service unavailable |
| Reranking | Model timeout | Skip reranking, use raw scores |
| LLM | Generation failure | Retry with exponential backoff |
| LLM | Timeout (>30s) | Return partial answer or timeout error |
| Guardrails | Hallucination detected | Return low-confidence warning |
| Overall | Any unhandled error | Log, return generic error, alert |

## Retry & Timeout Policy

```
Retry Configuration:
  - Max retries: 3 (configurable)
  - Backoff: Exponential (1s, 2s, 4s)
  - Timeout per stage:
      STT:           10s
      Embedding:      5s
      Vector Search:  2s
      Reranking:      5s
      LLM:           30s (configurable)
      Guardrails:     3s
  - Total pipeline timeout: 60s
```

## Logging Architecture

Every request generates a structured log trail:

```json
{
  "request_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "stage": "retrieval|generation|guardrails",
  "duration_ms": 42,
  "status": "success|error|timeout",
  "metadata": {}
}
```

## Latency Tracking

Latency is measured at every pipeline stage. Aggregated metrics:
- **P50**: Median latency (target: <200ms for retrieval)
- **P70**: 70th percentile
- **P100**: Maximum observed latency

See `docs/latency_strategy.md` for the full measurement plan.

## Fallback Behavior

```
IF STT fails → ask user to type query
IF retrieval returns 0 results → broaden search / inform user
IF LLM times out → return "processing" with async result
IF guardrails flag hallucination → return disclaimer
IF all retries exhausted → return structured error with trace ID
```

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Backend | Flask (Python) | Lightweight, production-tested |
| Dataset | MSMARCO-XI | Official HH Goa 2026 requirement |
| Vector DB | FAISS | Fast, in-memory, well-supported |
| Embeddings | HF Sentence Transformers | Open-source, benchmarkable |
| LLM | Configurable | Supports multiple providers |
| STT | Sarvam / ElevenLabs | Indic language support |
| Frontend | React + Vite + Tailwind | Modern, fast, professional |
| Testing | pytest | Standard Python testing |
