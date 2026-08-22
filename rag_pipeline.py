"""
rag_pipeline.py - Legacy and High-Level Interface for TTS-Cloud.
Delegates to the modern modular architecture (config, document_processor, retriever, rag_engine).
"""

from typing import Any, Dict, List, Optional, Tuple

from config import RAGSettings, get_api_keys
from document_processor import process_pdf_into_chunks
from retriever import SyllabusRetriever
from rag_engine import SyllabusRAGEngine, RAGResponse

# Global shared singleton instances
_retriever_instance: Optional[SyllabusRetriever] = None
_engine_instance: Optional[SyllabusRAGEngine] = None


def get_retriever() -> SyllabusRetriever:
    """Get or initialize the shared retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SyllabusRetriever()
    return _retriever_instance


def get_rag_engine() -> SyllabusRAGEngine:
    """Get or initialize the shared RAG engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SyllabusRAGEngine(retriever=get_retriever())
    return _engine_instance


def process_pdf(pdf_bytes: bytes, filename: str = "Syllabus.pdf") -> Tuple[str, int]:
    """
    Ingest a PDF, extract page-aware syllabus units, chunk, embed, and index in Pinecone.
    Returns:
        (document_id, num_chunks)
    """
    doc_id, chunks = process_pdf_into_chunks(pdf_bytes, filename=filename)
    retriever = get_retriever()
    num_indexed = retriever.index_syllabus_chunks(chunks)
    return doc_id, num_indexed


def answer_question(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    document_filter: Optional[str] = None
) -> Tuple[str, str, RAGResponse]:
    """
    High-level query function compatible with legacy calls while returning full RAG metadata.
    Returns:
        (answer_text, formatted_sources_str, full_rag_response_object)
    """
    engine = get_rag_engine()
    response = engine.answer_question(
        question=question,
        chat_history=chat_history,
        document_filter=document_filter
    )
    
    # Format source summary string for legacy compatibility
    source_items = []
    for c in response.citations:
        source_items.append(f"{c['filename']} (p. {c['page']} — {c['unit']})")
    sources_str = " | ".join(source_items) if source_items else "No direct citation available"
    
    return response.answer, sources_str, response
