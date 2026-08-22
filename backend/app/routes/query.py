import os
import requests
from flask import Blueprint, request, jsonify
from backend.app.config import settings
from backend.app.services.retrieval import RetrievalService
from backend.app.utils.logging import get_logger
from backend.app.services.latency import LatencyTracker

logger = get_logger("query_route")
query_bp = Blueprint("query", __name__)

# Lazy-loaded retrieval service instance
_retrieval_service = None

def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        logger.info("Initializing RetrievalService for API...")
        _retrieval_service = RetrievalService()
        try:
            # Load the existing sentence FAISS index
            _retrieval_service.load_index("sentence", base_path=str(settings.INDEXES_DIR))
        except FileNotFoundError as e:
            logger.error("FAISS index not found for sentence strategy: %s", e)
    return _retrieval_service


@query_bp.route("/api/query", methods=["POST"])
def run_query():
    """
    POST /api/query
    Payload: { "query": "What is Python?", "lang": "hi" }
    """
    tracker = LatencyTracker()
    tracker.start("api_total")

    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    lang = data.get("lang", "").strip()

    if not query_text:
        return jsonify({
            "error": "bad_request",
            "message": "Query parameter cannot be empty."
        }), 400

    # 1. FAISS Semantic Retrieval
    retrieval_service = get_retrieval_service()
    if retrieval_service.vector_store is None:
        return jsonify({
            "error": "service_unavailable",
            "message": "FAISS retrieval index is not built or available."
        }), 503

    logger.info("Processing RAG query: '%s'", query_text)
    
    try:
        tracker.start("retrieval")
        # Run retrieval using the tracker
        response = retrieval_service.retrieve(query_text, tracker=tracker)
        tracker.stop("retrieval")
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return jsonify({
            "error": "internal_server_error",
            "message": f"Failed to retrieve context: {str(e)}"
        }), 500

    # Extract retrieved texts as context
    retrieved_passages = [r.text for r in response.results]
    context = "\n\n".join(f"Passage {i+1}:\n{text}" for i, text in enumerate(retrieved_passages))

    # 2. Call Groq LLM (RAG Generation)
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY environment variable is not set.")
        return jsonify({
            "error": "configuration_error",
            "message": "LLM API key is not configured on the backend."
        }), 500

    tracker.start("llm")
    # Call Groq API via request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are an intelligent Indic language Voice Assistant for Hacker House Goa. "
        "Answer the user query based ONLY on the provided context passages. "
        "Strictly ground your answers. If the answer cannot be found in the context, "
        "clearly state 'I do not know the answer based on my knowledge base.' "
        "Do not hallucinate or make up facts."
    )
    if lang:
        system_prompt += f" Answer the user query in the requested language (language code: {lang})."

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context passages:\n{context}\n\nQuery: {query_text}"}
        ],
        "temperature": 0.1
    }

    try:
        groq_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.LLM_TIMEOUT
        )
        if not groq_resp.ok:
            logger.error("Groq API error response details: %s", groq_resp.text)
        groq_resp.raise_for_status()
        answer = groq_resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        # Fallback to pure retrieval text in case Groq is down or key is invalid
        answer = (
            f"[Grounded LLM Generation Failed: {str(e)}]\n\n"
            "Here is the retrieved context from the FAISS database:\n" + context
        )
    finally:
        if "llm" in tracker._start_times:
            tracker.stop("llm")

    tracker.stop("api_total")

    # Construct final RAG response
    result = {
        "query": query_text,
        "answer": answer,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": round(r.score, 6),
                "rank": r.rank,
                "metadata": r.metadata
            }
            for r in response.results
        ],
        "latency": {
            "embedding_ms": round(tracker.elapsed("embedding"), 2),
            "search_ms": round(tracker.elapsed("faiss_search"), 2),
            "llm_ms": round(tracker.elapsed("llm"), 2),
            "total_ms": round(tracker.elapsed("api_total"), 2)
        }
    }
    
    logger.info(
        "RAG complete | total_latency=%.2fms | retrieved_chunks=%d", 
        result["latency"]["total_ms"], 
        len(result["results"])
    )
    return jsonify(result), 200
