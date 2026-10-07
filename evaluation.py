"""
evaluation.py - Lightweight Evaluation Framework for TTS-Cloud RAG.
Measures Retrieval Quality (Hit@K, MRR), Groundedness / Context Overlap, and Latency
using an academic syllabus benchmark dataset.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import RAGSettings, setup_logging
from document_processor import SyllabusChunk
from retriever import SyllabusRetriever
from rag_engine import SyllabusRAGEngine

logger = setup_logging()


# ─── Benchmark Evaluation Dataset (Academic Syllabus QA) ──────────────────────
# Ground truth sample dataset covering core Computer Science syllabus domains
BENCHMARK_EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "id": "eval_dbms_01",
        "subject": "Database Management Systems",
        "question": "What is Third Normal Form (3NF) and what condition must hold for functional dependencies?",
        "expected_keywords": ["3NF", "Third Normal Form", "transitive dependency", "superkey", "prime attribute"],
        "expected_unit": "Unit 3",
        "expected_page_hint": 14,
        "sample_context": (
            "[DBMS.pdf | Page 14 | Unit 3: Normalization | 3.3 3NF]\n"
            "A relation R is in Third Normal Form (3NF) if for every non-trivial functional dependency X -> A, "
            "either X is a superkey of R, or A is a prime attribute (part of some candidate key). "
            "This eliminates transitive dependencies."
        )
    },
    {
        "id": "eval_cn_02",
        "subject": "Computer Networks",
        "question": "What are the key differences between TCP and UDP at the transport layer?",
        "expected_keywords": ["TCP", "UDP", "connection-oriented", "connectionless", "reliable", "unreliable", "handshake"],
        "expected_unit": "Unit 4",
        "expected_page_hint": 22,
        "sample_context": (
            "[ComputerNetworks.pdf | Page 22 | Unit 4: Transport Layer | 4.2 TCP vs UDP]\n"
            "TCP is a connection-oriented, reliable byte-stream protocol with three-way handshaking, congestion control, "
            "and flow control. UDP is a connectionless, lightweight, unreliable datagram protocol without handshakes."
        )
    },
    {
        "id": "eval_os_03",
        "subject": "Operating Systems",
        "question": "What are the four necessary conditions for a deadlock to occur?",
        "expected_keywords": ["mutual exclusion", "hold and wait", "no preemption", "circular wait", "deadlock"],
        "expected_unit": "Unit 2",
        "expected_page_hint": 8,
        "sample_context": (
            "[OperatingSystems.pdf | Page 8 | Unit 2: Process Synchronization & Deadlocks | 2.4 Deadlock Conditions]\n"
            "The four Coffman conditions for deadlock are: 1. Mutual Exclusion, 2. Hold and Wait, "
            "3. No Preemption, and 4. Circular Wait. All four must hold simultaneously."
        )
    },
    {
        "id": "eval_daa_04",
        "subject": "Design and Analysis of Algorithms",
        "question": "What is the time complexity of Dijkstra's shortest path algorithm using a min-priority queue?",
        "expected_keywords": ["Dijkstra", "O((V + E) log V)", "priority queue", "shortest path", "greedy"],
        "expected_unit": "Unit 3",
        "expected_page_hint": 18,
        "sample_context": (
            "[DAA.pdf | Page 18 | Unit 3: Greedy Algorithms & Graphs | 3.5 Dijkstra]\n"
            "Dijkstra's single-source shortest path algorithm using a binary min-heap / priority queue "
            "runs in O((V + E) log V) time, where V is vertices and E is edges. It requires non-negative edge weights."
        )
    },
    {
        "id": "eval_oos_05",
        "subject": "Out-of-Syllabus / Guardrail Check",
        "question": "What is the capital city of Australia and what is its population?",
        "expected_keywords": ["not available", "not found", "syllabus", "not appear"],
        "expected_unit": "Out-of-Scope",
        "expected_page_hint": -1,
        "sample_context": ""
    }
]


@dataclass
class EvalResult:
    """Individual test sample evaluation metrics."""
    test_id: str
    question: str
    hit_at_k: bool
    reciprocal_rank: float
    keyword_coverage: float
    groundedness_score: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    confidence_score: float


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark metrics."""
    total_samples: int
    mean_hit_at_k: float
    mean_mrr: float
    mean_keyword_coverage: float
    mean_groundedness: float
    avg_retrieval_latency_ms: float
    avg_total_latency_ms: float
    detailed_results: List[Dict[str, Any]]


class RAGEvaluator:
    """
    Evaluator to test and benchmark the TTS-Cloud RAG Pipeline.
    Supports in-memory synthetic indexing or live Pinecone benchmarking.
    """

    def __init__(self, engine: Optional[SyllabusRAGEngine] = None):
        self.engine = engine or SyllabusRAGEngine()
        self.retriever = self.engine.retriever

    def seed_synthetic_benchmark_corpus(self) -> int:
        """Seed the retriever's BM25 and in-memory cache with benchmark samples for offline testing."""
        synthetic_chunks: List[SyllabusChunk] = []
        for i, item in enumerate(BENCHMARK_EVAL_DATASET):
            if not item["sample_context"]:
                continue
            chunk = SyllabusChunk(
                chunk_id=f"benchmark_doc_{i}",
                document_id="benchmark_suite",
                filename=f"{item['subject'].replace(' ', '_')}.pdf",
                page=item["expected_page_hint"],
                text=item["sample_context"],
                unit=item["expected_unit"],
                section="Benchmark Topic",
                char_count=len(item["sample_context"])
            )
            synthetic_chunks.append(chunk)

        # Index chunks into retriever cache and build BM25
        self.retriever.index_syllabus_chunks(synthetic_chunks)
        return len(synthetic_chunks)

    def evaluate_sample(self, sample: Dict[str, Any], top_k: int = 4) -> EvalResult:
        """Run evaluation on a single benchmark query."""
        q = sample["question"]
        expected_kw = [kw.lower() for kw in sample["expected_keywords"]]
        is_oos = sample.get("expected_page_hint", 0) == -1

        t0 = time.time()
        # Retrieve candidate chunks
        docs, confidence = self.retriever.hybrid_search(
            query=q,
            top_k=top_k,
            candidate_pool_size=8
        )
        retrieval_latency = round((time.time() - t0) * 1000, 2)

        # 1. Retrieval Hit@K & Reciprocal Rank
        hit = False
        rr = 0.0

        if not is_oos:
            for rank, doc in enumerate(docs, start=1):
                doc_text_lower = doc.text.lower()
                matches = [kw for kw in expected_kw if kw in doc_text_lower]
                if matches and not hit:
                    hit = True
                    rr = 1.0 / rank
        else:
            # For out-of-scope query, success means low confidence guardrail fires or no relevant docs
            if confidence < self.retriever.settings.CONFIDENCE_THRESHOLD or not docs:
                hit = True
                rr = 1.0

        # 2. Answer generation & Groundedness evaluation
        t1 = time.time()
        generation_latency = 0.0
        kw_cov = 0.0
        groundedness = 0.0

        try:
            rag_resp = self.engine.answer_question(question=q, top_k=top_k)
            answer = rag_resp.answer
            generation_latency = round((time.time() - t1) * 1000, 2)

            ans_lower = answer.lower()
            matched_kws = [kw for kw in expected_kw if kw in ans_lower]
            kw_cov = len(matched_kws) / len(expected_kw) if expected_kw else 1.0

            if is_oos:
                # If answer correctly flags that information is unavailable, consider it a hit
                if any(kw in ans_lower for kw in expected_kw):
                    hit = True
                    rr = 1.0
                    groundedness = 1.0
            else:
                context_text = " ".join([d.text for d in docs]).lower()
                ans_words = set(ans_lower.split())
                ctx_words = set(context_text.split())
                if ans_words and ctx_words:
                    overlap = len(ans_words.intersection(ctx_words))
                    groundedness = min(1.0, overlap / max(1, len(ans_words)))
        except Exception as e:
            logger.error("Generation error during evaluation of '%s': %s", sample["id"], e)
            generation_latency = 0.0
            kw_cov = 0.0
            groundedness = 0.0

        return EvalResult(
            test_id=sample["id"],
            question=q,
            hit_at_k=hit,
            reciprocal_rank=round(rr, 3),
            keyword_coverage=round(kw_cov, 3),
            groundedness_score=round(groundedness, 3),
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            confidence_score=round(confidence, 3)
        )

    def run_benchmark(self, top_k: int = 4) -> BenchmarkSummary:
        """Run all test cases in the benchmark suite and compute aggregated metrics."""
        self.seed_synthetic_benchmark_corpus()
        results: List[EvalResult] = []

        for sample in BENCHMARK_EVAL_DATASET:
            res = self.evaluate_sample(sample, top_k=top_k)
            results.append(res)

        total = len(results)
        mean_hit = sum(1 for r in results if r.hit_at_k) / total if total > 0 else 0.0
        mean_mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0.0
        mean_kw = sum(r.keyword_coverage for r in results) / total if total > 0 else 0.0
        mean_grounded = sum(r.groundedness_score for r in results) / total if total > 0 else 0.0
        avg_ret_lat = sum(r.retrieval_latency_ms for r in results) / total if total > 0 else 0.0
        avg_tot_lat = sum(r.retrieval_latency_ms + r.generation_latency_ms for r in results) / total if total > 0 else 0.0

        return BenchmarkSummary(
            total_samples=total,
            mean_hit_at_k=round(mean_hit, 3),
            mean_mrr=round(mean_mrr, 3),
            mean_keyword_coverage=round(mean_kw, 3),
            mean_groundedness=round(mean_grounded, 3),
            avg_retrieval_latency_ms=round(avg_ret_lat, 2),
            avg_total_latency_ms=round(avg_tot_lat, 2),
            detailed_results=[asdict(r) for r in results]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS-Cloud RAG Benchmark Evaluator")
    parser.add_argument("--top-k", type=int, default=4, help="Top-K retrieval depth (default: 4)")
    parser.add_argument("--assert-min-hit", type=float, default=None, help="Assert minimum hit rate (e.g. 0.8) for CI test gates")
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 RUNNING TTS-CLOUD RAG BENCHMARK EVALUATION SUITE")
    print("=" * 65)

    evaluator = RAGEvaluator()
    summary = evaluator.run_benchmark(top_k=args.top_k)

    print(f"\n📊 Evaluated {summary.total_samples} Benchmark Queries:")
    print(f"  • Hit Rate @ Top-{args.top_k}:       {summary.mean_hit_at_k * 100:.1f}%")
    print(f"  • Mean Reciprocal Rank:   {summary.mean_mrr:.3f}")
    print(f"  • Keyword Coverage:       {summary.mean_keyword_coverage * 100:.1f}%")
    print(f"  • Groundedness / Overlap: {summary.mean_groundedness * 100:.1f}%")
    print(f"  • Avg Retrieval Latency:  {summary.avg_retrieval_latency_ms:.1f} ms")
    print(f"  • Avg Total End-to-End:   {summary.avg_total_latency_ms:.1f} ms")
    print("\nDetailed Case Results:")
    for r in summary.detailed_results:
        print(f"  • [{r['test_id']}] Hit: {r['hit_at_k']} | MRR: {r['reciprocal_rank']} | Latency: {r['retrieval_latency_ms']}ms | Conf: {r['confidence_score']}")
    print("=" * 65)

    if args.assert_min_hit is not None:
        if summary.mean_hit_at_k < args.assert_min_hit:
            print(f"❌ CI Gate Failed: Hit rate {summary.mean_hit_at_k:.2f} is below threshold {args.assert_min_hit:.2f}")
            sys.exit(1)
        else:
            print(f"✅ CI Gate Passed: Hit rate {summary.mean_hit_at_k:.2f} >= {args.assert_min_hit:.2f}")
