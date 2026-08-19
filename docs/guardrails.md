# Guardrails Design — HH Goa 2026 Voice-Enabled RAG

> **Status**: Design Document (Step 1 — Implementation in Step 3)
> **Last Updated**: 2026-08-19

---

## Overview

Guardrails protect the system from producing unsafe, inaccurate, or inappropriate responses. They operate at both input (pre-generation) and output (post-generation) stages.

## Guardrail Categories

### 1. Off-Topic Detection (Input Guardrail)

**Problem**: Users may ask questions unrelated to the available knowledge base.

**Strategy**:
```
1. Classify query intent using keyword/embedding matching
2. Check if retrieved passages have minimum relevance score
3. If top retrieval score < threshold → flag as off-topic
```

**Response**:
```json
{
  "answer": "This question appears to be outside the scope of the available knowledge base. I can help with questions related to the topics covered in the MSMARCO dataset.",
  "status": "off_topic",
  "confidence": 0.0
}
```

**Thresholds**:
- Relevance score < 0.3 → likely off-topic
- Relevance score 0.3–0.5 → uncertain, answer with disclaimer
- Relevance score > 0.5 → on-topic

---

### 2. Missing Context (Retrieval Guardrail)

**Problem**: Retrieval may not find enough evidence to answer the question.

**Strategy**:
```
1. Count retrieved passages above relevance threshold
2. If fewer than min_relevant_passages → insufficient context
3. Check if context covers the query's key entities
```

**Response**:
```json
{
  "answer": "I couldn't find enough relevant information in the available knowledge to answer this reliably.",
  "status": "insufficient_context",
  "confidence": 0.15,
  "suggestion": "Try rephrasing your question or asking about a related topic."
}
```

**Configuration**:
- `min_relevant_passages`: 2
- `min_relevance_score`: 0.4
- `min_entity_coverage`: 0.5

---

### 3. Unsafe Input Handling (Input Guardrail)

**Problem**: Users may submit inappropriate, harmful, or adversarial inputs.

**Strategy**:
```
1. Keyword blocklist check (fast, first-pass)
2. Pattern matching for injection attempts
3. Content classification (if model available)
4. Reject or sanitize unsafe inputs
```

**Categories**:
| Category | Action |
|----------|--------|
| Profanity / hate speech | Reject with message |
| Prompt injection attempts | Sanitize and log |
| Personal data (PII) | Redact and warn |
| Extremely long inputs | Truncate to max length |
| Empty / gibberish | Return validation error |

**Response**:
```json
{
  "answer": "I'm unable to process this request. Please rephrase your question appropriately.",
  "status": "rejected",
  "reason": "unsafe_input"
}
```

---

### 4. Hallucination Detection (Output Guardrail)

**Problem**: LLMs may generate plausible-sounding answers that are not supported by the retrieved context.

**Strategy**:

#### Level 1: Overlap Check (Fast)
```
- Extract key claims from generated answer
- Check if each claim appears in / is supported by context
- Grounding score = supported_claims / total_claims
```

#### Level 2: Entailment Check (Thorough)
```
- Use NLI model to check if context entails each claim
- Classify as: ENTAILMENT, NEUTRAL, CONTRADICTION
- Flag CONTRADICTION claims
```

#### Level 3: Citation Verification
```
- If answer cites source passages
- Verify cited passages actually support the claim
- Flag unsupported citations
```

**Thresholds**:
| Grounding Score | Action |
|----------------|--------|
| > 0.8 | Pass — high confidence |
| 0.5 – 0.8 | Pass with disclaimer |
| 0.3 – 0.5 | Return with strong warning |
| < 0.3 | Reject — likely hallucination |

**Response (low grounding)**:
```json
{
  "answer": "Based on the available information: [answer]. Note: This response has lower confidence as the available evidence only partially supports this answer.",
  "status": "low_confidence",
  "grounding_score": 0.42,
  "warnings": ["Low grounding score — answer may not be fully supported by evidence"]
}
```

---

## Guardrail Pipeline

```
         Input
           │
           ▼
    ┌──────────────┐
    │ Input Safety  │──→ REJECT (if unsafe)
    │ Check         │
    └──────┬───────┘
           │ (safe)
           ▼
    ┌──────────────┐
    │ Off-Topic     │──→ OFF-TOPIC response
    │ Detection     │
    └──────┬───────┘
           │ (on-topic)
           ▼
    ┌──────────────┐
    │ Context       │──→ INSUFFICIENT_CONTEXT response
    │ Sufficiency   │
    └──────┬───────┘
           │ (sufficient)
           ▼
      [LLM Generation]
           │
           ▼
    ┌──────────────┐
    │ Hallucination │──→ WARNING or REJECT
    │ Check         │
    └──────┬───────┘
           │ (grounded)
           ▼
    ┌──────────────┐
    │ Output Safety │──→ REDACT (if unsafe content generated)
    │ Check         │
    └──────┬───────┘
           │ (safe)
           ▼
      Final Answer
```

## Configuration

```python
@dataclass
class GuardrailConfig:
    # Off-topic
    off_topic_threshold: float = 0.3
    uncertain_threshold: float = 0.5

    # Context sufficiency
    min_relevant_passages: int = 2
    min_relevance_score: float = 0.4

    # Hallucination
    grounding_high: float = 0.8
    grounding_medium: float = 0.5
    grounding_reject: float = 0.3

    # Safety
    max_input_length: int = 1000  # characters
    enable_blocklist: bool = True
    enable_pii_detection: bool = True

    # Behavior
    strict_mode: bool = False  # If True, reject on any warning
```

## Monitoring & Logging

Every guardrail check is logged:

```json
{
  "trace_id": "...",
  "guardrail": "hallucination_check",
  "result": "pass",
  "score": 0.87,
  "duration_ms": 12,
  "details": {
    "claims_checked": 4,
    "claims_supported": 3,
    "claims_unsupported": 1
  }
}
```

Aggregate metrics tracked:
- % of queries flagged as off-topic
- % of queries with insufficient context
- % of answers with low grounding score
- % of inputs rejected for safety
- Average guardrail latency
