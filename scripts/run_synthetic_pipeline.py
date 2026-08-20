"""
Synthetic Pipeline Runner
==========================
Runs the full preprocessing → chunking → embedding → FAISS → retrieval
pipeline using synthetic MSMARCO-XI-shaped data.

This bypasses the Arrow streaming limitation while exercising
all real pipeline components with actual embedding and FAISS operations.

Produces real measurements for the Step 2 final report.
"""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.preprocessing import PreprocessingPipeline, PreprocessingConfig
from backend.app.services.chunking import create_chunker, STRATEGY_REGISTRY
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.services.retrieval import RetrievalService
from backend.app.config import settings


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def make_synthetic_records(n: int) -> list[dict]:
    """
    Create n synthetic MSMARCO-XI-shaped records with realistic passages.
    Passages are real English text matching the dataset structure.
    """
    query_types = ["DESCRIPTION", "NUMERIC", "ENTITY", "LOCATION", "PERSON"]
    passages_pool = [
        "Python is a high-level, general-purpose programming language known for its readable syntax and dynamic typing. It supports multiple programming paradigms including object-oriented, functional, and procedural styles.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed, focusing on the development of computer programs that can access data and use it to learn.",
        "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based models by first retrieving relevant documents from a knowledge base and then using a language model to generate a response conditioned on the retrieved content.",
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
        "FAISS (Facebook AI Similarity Search) is an open-source library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors, capable of searching in sets of vectors of any size.",
        "Vector embeddings are numerical representations of data (text, images, etc.) in a high-dimensional space where similar items are located near each other, enabling semantic similarity search.",
        "The transformer architecture, introduced in the paper Attention Is All You Need, revolutionized natural language processing by using self-attention mechanisms instead of recurrent neural networks.",
        "Sentence transformers are specialized models that map sentences and paragraphs to a dense vector space and can be used for tasks such as clustering or semantic search using cosine similarity.",
        "The MS MARCO dataset is a large-scale machine reading comprehension and question answering dataset created by Microsoft, containing questions from Bing search queries with human-generated answers.",
        "Information retrieval is the process of obtaining information system resources relevant to an information need from a collection of resources, evaluating and ranking documents based on relevance to a query.",
        "Dense retrieval uses neural network encoders to map queries and documents into a shared dense vector space, then retrieves documents by computing similarity in that space.",
        "Sparse retrieval methods like BM25 use term frequency and inverse document frequency to compute lexical similarity between queries and documents.",
        "Reranking is the process of taking an initial list of retrieved documents and reordering them using a more powerful but computationally expensive model.",
        "Chunking is the process of splitting long documents into smaller pieces that fit within the context window of an embedding or language model.",
        "The MSMARCO-XI dataset extends MS MARCO with translations into 14 Indic languages including Hindi, Bengali, Tamil, Telugu, and Urdu.",
        "Indic languages include a large family of languages spoken in the Indian subcontinent such as Hindi, Bengali, Punjabi, Gujarati, Marathi, and many others.",
        "Cross-lingual information retrieval involves retrieving documents in one language using queries in a different language, requiring multilingual embeddings.",
        "The paraphrase-multilingual-MiniLM-L12-v2 model maps sentences from 50+ languages into a shared multilingual embedding space for cross-lingual semantic search.",
        "Evaluation metrics for information retrieval include Recall at K, Precision at K, Mean Reciprocal Rank, and Normalized Discounted Cumulative Gain.",
        "Latency in information retrieval systems is typically measured at the P50, P70, P95, and P99 percentiles to understand both typical and worst-case performance.",
    ]

    records = []
    for i in range(n):
        # Each record has 5-10 passages
        num_passages = (i % 6) + 5
        selected_passages = [passages_pool[(i + j) % len(passages_pool)] for j in range(num_passages)]
        is_selected = [1 if j == (i % num_passages) else 0 for j in range(num_passages)]

        records.append({
            "query_id": i + 1,
            "query": f"synthetic translated query {i}",
            "Eng_Query": f"What is {['python', 'machine learning', 'RAG', 'NLP', 'FAISS'][i % 5]}?",
            "Answer": f"synthetic answer {i}",
            "Eng_Answer": f"English answer for query {i}",
            "query_type": query_types[i % len(query_types)],
            "source_lang": "en",
            "target_lang": ["hi", "bn", "gu", "ta", "te"][i % 5],
            "meta": {"model_name": "synthetic", "temperature": 0.7, "max_tokens": 512, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
            "passages": {
                "English_passages": selected_passages,
                "Translated_passages": [f"Translation of: {p[:50]}" for p in selected_passages],
                "is_selected": is_selected,
            },
        })
    return records


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def run_pipeline(n_records: int = 200, strategy: str = "sentence"):
    """Run the complete pipeline and return measured results."""
    safe_print(f"\n{'='*70}")
    safe_print(f"  Synthetic Pipeline Run")
    safe_print(f"  Records: {n_records} | Strategy: {strategy}")
    safe_print(f"{'='*70}\n")

    results = {"strategy": strategy, "n_records": n_records}

    # 1. Synthetic data
    safe_print("[1/6] Generating synthetic data...")
    records = make_synthetic_records(n_records)
    safe_print(f"  Generated {len(records)} records\n")

    # 2. Preprocess
    safe_print("[2/6] Preprocessing...")
    t0 = time.time()
    pipeline = PreprocessingPipeline()
    documents = pipeline.process_records(records)
    prep_time = time.time() - t0
    stats = pipeline.get_stats()
    results["preprocessing"] = {
        "documents": len(documents),
        "time_s": round(prep_time, 3),
        "stats": stats,
    }
    safe_print(f"  {len(documents)} documents in {prep_time:.3f}s\n")

    # 3. Chunk with each strategy
    safe_print("[3/6] Chunking...")
    chunk_results = {}
    for strat in ["fixed", "overlap", "sentence", "metadata"]:
        kwargs = {"chunk_size": 512}
        if strat == "overlap":
            kwargs["overlap"] = 64
        chunker = create_chunker(strat, **kwargs)
        t0 = time.time()
        chunks = chunker.chunk_batch(documents)
        elapsed = time.time() - t0
        lengths = [c.char_count for c in chunks]
        sorted_lengths = sorted(lengths) if lengths else [0]
        n = len(sorted_lengths)
        chunk_results[strat] = {
            "num_chunks": len(chunks),
            "time_s": round(elapsed, 4),
            "avg_chars": round(sum(lengths) / n, 1) if n else 0,
            "median_chars": sorted_lengths[n // 2] if n else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }
        safe_print(f"  {strat:12s}: {len(chunks):4d} chunks | avg {chunk_results[strat]['avg_chars']:6.0f} chars | {elapsed:.4f}s")
    results["chunking"] = chunk_results
    safe_print()

    # 4. Use sentence chunks for embedding/retrieval
    safe_print("[4/6] Embedding (sentence chunks)...")
    chunker = create_chunker("sentence", chunk_size=512)
    chunks = chunker.chunk_batch(documents)
    texts = [c.text for c in chunks]

    emb_service = EmbeddingService()
    t0 = time.time()
    embeddings = emb_service.embed_batch(texts, show_progress=True)
    emb_time = time.time() - t0
    throughput = round(len(texts) / emb_time, 1)
    results["embeddings"] = {
        "model": emb_service.model_name,
        "dimension": embeddings.shape[1],
        "num_texts": len(texts),
        "time_s": round(emb_time, 2),
        "throughput_texts_per_sec": throughput,
    }
    safe_print(f"  {len(texts)} texts in {emb_time:.2f}s ({throughput} texts/sec)")
    safe_print(f"  Dimension: {embeddings.shape[1]}\n")

    # 5. Build FAISS index
    safe_print("[5/6] Building FAISS index...")
    index_dir = settings.INDEXES_DIR / "sentence"
    t0 = time.time()
    store = VectorStore(
        dimension=embeddings.shape[1],
        index_type="IndexFlatIP",
        base_path=str(settings.INDEXES_DIR),
    )
    store.add_chunks(chunks, embeddings)
    store.save("sentence")
    idx_time = time.time() - t0
    results["faiss"] = {
        "index_type": "IndexFlatIP",
        "total_vectors": store.size,
        "dimension": embeddings.shape[1],
        "time_s": round(idx_time, 3),
    }
    safe_print(f"  Indexed {store.size} vectors in {idx_time:.3f}s\n")

    # 6. Retrieval benchmark
    safe_print("[6/6] Retrieval latency benchmark...")
    queries = list(set(
        doc.metadata.get("eng_query", "")
        for doc in documents
        if doc.metadata.get("eng_query", "").strip()
    ))
    if not queries:
        queries = [f"what is query {i}" for i in range(50)]

    retrieval = RetrievalService(
        embedding_service=emb_service,
        vector_store=store,
        top_k=5,
    )

    # Warmup
    for q in queries[:3]:
        retrieval.retrieve(q)

    # Measure
    embed_lats, search_lats, total_lats = [], [], []
    for q in queries:
        resp = retrieval.retrieve(q)
        embed_lats.append(resp.embedding_latency_ms)
        search_lats.append(resp.search_latency_ms)
        total_lats.append(resp.total_latency_ms)

    results["retrieval_latency"] = {
        "num_queries": len(total_lats),
        "embedding_ms": {
            "p50": round(percentile(embed_lats, 50), 2),
            "p70": round(percentile(embed_lats, 70), 2),
            "p95": round(percentile(embed_lats, 95), 2),
            "p100": round(max(embed_lats), 2),
        },
        "faiss_search_ms": {
            "p50": round(percentile(search_lats, 50), 2),
            "p70": round(percentile(search_lats, 70), 2),
            "p95": round(percentile(search_lats, 95), 2),
            "p100": round(max(search_lats), 2),
        },
        "total_ms": {
            "p50": round(percentile(total_lats, 50), 2),
            "p70": round(percentile(total_lats, 70), 2),
            "p95": round(percentile(total_lats, 95), 2),
            "p100": round(max(total_lats), 2),
        },
    }

    lat = results["retrieval_latency"]["total_ms"]
    safe_print(f"  {len(total_lats)} queries | P50: {lat['p50']}ms | P70: {lat['p70']}ms | P100: {lat['p100']}ms\n")

    # Save
    out_path = project_root / "evaluation" / "results" / "synthetic_pipeline_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    safe_print(f"{'='*70}")
    safe_print(f"  Pipeline Complete! Results: {out_path}")
    safe_print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    results = run_pipeline(n_records=200, strategy="sentence")

    # Print summary
    safe_print("\nCHUNKING COMPARISON:")
    safe_print(f"  {'Strategy':<12} {'Chunks':>7} {'Avg Chars':>10} {'Time (s)':>10}")
    for strat, r in results["chunking"].items():
        safe_print(f"  {strat:<12} {r['num_chunks']:>7} {r['avg_chars']:>10.0f} {r['time_s']:>10.4f}")

    safe_print("\nRETRIEVAL LATENCY (sentence strategy):")
    lat = results["retrieval_latency"]["total_ms"]
    safe_print(f"  P50:  {lat['p50']}ms")
    safe_print(f"  P70:  {lat['p70']}ms")
    safe_print(f"  P100: {lat['p100']}ms")
