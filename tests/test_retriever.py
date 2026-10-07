"""
test_retriever.py - Unit tests for custom BM25Okapi, Reciprocal Rank Fusion (RRF),
score blending, and local vector retrieval fallback.
"""

import pytest
from retriever import BM25Okapi, SyllabusRetriever, RetrievedDocument
from document_processor import SyllabusChunk


def test_bm25_tokenization_preserves_technical_tokens():
    bm25 = BM25Okapi([])
    tokens = bm25._tokenize("Study 3NF, BCNF, TCP/IP and O(V+E) algorithms.")
    assert "3nf" in tokens
    assert "bcnf" in tokens
    assert "tcp/ip" in tokens


def test_bm25_scoring_exact_match():
    corpus = [
        "Relational database management systems and Boyce Codd Normal Form 3NF BCNF.",
        "Computer networks OSI model physical layer ethernet cables.",
        "Operating systems process scheduling deadlocks semaphore mutual exclusion."
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores("BCNF Normal Form")

    # Document 0 should receive the highest score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_retriever_in_memory_index_and_search():
    # Use empty pinecone key to verify in-memory vector cache and BM25 search
    retriever = SyllabusRetriever(pinecone_api_key="")

    chunks = [
        SyllabusChunk(
            chunk_id="test_c1",
            document_id="test_doc",
            filename="DBMS.pdf",
            page=14,
            text="[DBMS.pdf | Page 14 | Unit 3 | 3.3 3NF]\nThird Normal Form functional dependency superkey.",
            unit="Unit 3",
            section="3.3 3NF",
            char_count=90
        ),
        SyllabusChunk(
            chunk_id="test_c2",
            document_id="test_doc",
            filename="Networks.pdf",
            page=22,
            text="[Networks.pdf | Page 22 | Unit 4 | 4.2 TCP]\nTransmission Control Protocol TCP handshake sequence numbers.",
            unit="Unit 4",
            section="4.2 TCP",
            char_count=98
        )
    ]

    retriever.index_syllabus_chunks(chunks)
    assert len(retriever._corpus_cache) >= 2

    # Query for DBMS topic
    docs, confidence = retriever.hybrid_search("functional dependency 3NF", top_k=2)
    assert len(docs) > 0
    assert docs[0].unit == "Unit 3"
    assert confidence > 0.0


def test_retriever_document_deletion():
    retriever = SyllabusRetriever(pinecone_api_key="")
    chunk = SyllabusChunk(
        chunk_id="del_c1",
        document_id="to_delete_doc",
        filename="Test.pdf",
        page=1,
        text="Sample syllabus text to delete.",
        unit="Unit 1",
        section="1.1",
        char_count=35
    )
    retriever.index_syllabus_chunks([chunk])
    assert any(c["document_id"] == "to_delete_doc" for c in retriever._corpus_cache)

    retriever.delete_document("to_delete_doc")
    assert not any(c["document_id"] == "to_delete_doc" for c in retriever._corpus_cache)
