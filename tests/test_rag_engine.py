"""
test_rag_engine.py - Unit tests for conversational query reformulation,
prompt injection guardrails, and grounded response formatting.
"""

import pytest
from rag_engine import SyllabusRAGEngine, RAGResponse
from retriever import SyllabusRetriever


def test_guardrail_triggers_on_empty_docs():
    retriever = SyllabusRetriever()
    # Mock hybrid search to simulate no matches found
    retriever.hybrid_search = lambda query, document_filter=None, top_k=4, candidate_pool_size=8: ([], 0.0)

    engine = SyllabusRAGEngine(retriever=retriever)
    resp: RAGResponse = engine.answer_question("What is quantum computing?")

    # Should trigger low-confidence guardrail immediately
    assert resp.is_low_confidence is True
    assert "No matching information found" in resp.answer or "not available" in resp.answer.lower()
    assert len(resp.retrieved_documents) == 0


def test_rag_response_dataclass_fields():
    resp = RAGResponse(
        answer="3NF eliminates transitive dependencies.",
        standalone_query="3NF transitive dependency",
        retrieved_documents=[],
        citations=[{"source_id": 1, "filename": "DBMS.pdf", "page": 14, "unit": "Unit 3", "section": "3.3", "score": 0.85, "snippet": "..."}],
        confidence_score=0.85,
        is_low_confidence=False,
        latency_ms=350.2
    )

    assert resp.answer.startswith("3NF")
    assert resp.confidence_score == 0.85
    assert len(resp.citations) == 1
    assert resp.latency_ms > 0
