"""
api.py - Production FastAPI REST API Server for Talk-to-Syllabus (TTS-Cloud).
Exposes endpoints for document ingestion, grounded academic querying,
study tool generation, and health diagnostics with strict Pydantic validation.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import RAGSettings, get_api_keys, get_cors_config, get_max_upload_bytes, setup_logging
from document_processor import process_pdf_into_chunks
from retriever import SyllabusRetriever
from rag_engine import SyllabusRAGEngine, RAGResponse

logger = setup_logging()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Reload the in-memory cache and BM25 index from Pinecone so a restart does not lose lexical search."""
    try:
        loaded = _retriever.ensure_hydrated()
        logger.info("Startup hydration finished (%d chunks loaded from the vector store).", loaded)
    except Exception as e:
        logger.warning("Startup hydration failed (%s); it will be retried on the next search.", e)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="TTS-Cloud API (Talk-to-Syllabus)",
    description=(
        "Production-grade, domain-specific Retrieval-Augmented Generation (RAG) API "
        "engineered for university syllabi, curriculum documents, and academic study assistance."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Multipart framing adds a little overhead on top of the file itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _upload_too_large(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"File too large. Maximum upload size is {limit} bytes (MAX_UPLOAD_MB)."},
    )


@app.middleware("http")
async def limit_ingest_body_size(request: Request, call_next):
    """Reject oversized ingest requests from Content-Length before the body is parsed.
    Chunked uploads without Content-Length are checked in the endpoint instead."""
    if request.method == "POST" and request.url.path == "/api/v1/ingest":
        limit = get_max_upload_bytes()
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > limit + _MULTIPART_OVERHEAD_BYTES:
            return _upload_too_large(limit)
    return await call_next(request)


# Configure CORS (added after the size middleware so it is the outermost layer and also decorates 413s).
# Origins come from CORS_ALLOW_ORIGINS; credentials are never combined with the "*" wildcard.
_cors_origins, _cors_credentials = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singleton instances
_retriever = SyllabusRetriever()
_engine = SyllabusRAGEngine(retriever=_retriever)



# ─── Pydantic Request & Response Schemas ─────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$", description="Role of the sender")
    content: str = Field(..., min_length=1, max_length=2000, description="Message text")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Student's query about the syllabus")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation turns")
    document_filter: Optional[str] = Field(default=None, description="Optional document_id to scope retrieval")
    top_k: int = Field(default=4, ge=1, le=12, description="Number of candidate chunks to retrieve")


class CitationItem(BaseModel):
    source_id: int
    filename: str
    page: int
    unit: str
    section: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    standalone_query: str
    confidence_score: float
    is_low_confidence: bool
    citations: List[CitationItem]
    latency_ms: float


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    detected_units: List[str]
    message: str


class UnitSummaryRequest(BaseModel):
    unit_name_or_query: str = Field(..., min_length=2, max_length=200)
    document_filter: Optional[str] = None


class UnitSummaryResponse(BaseModel):
    unit_query: str
    summary: str


class MCQRequest(BaseModel):
    topic_or_unit: str = Field(..., min_length=2, max_length=200)
    num_questions: int = Field(default=4, ge=1, le=10)
    document_filter: Optional[str] = None


class MCQResponse(BaseModel):
    topic: str
    mcqs: str


class ExamTopicsRequest(BaseModel):
    document_filter: Optional[str] = None


class ExamTopicsResponse(BaseModel):
    strategy_guide: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    groq_model: str
    embedding_model: str
    pinecone_index: str
    total_indexed_chunks: int
    uptime_seconds: float


_start_time = time.time()


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """System health check and diagnostic metrics."""
    return HealthResponse(
        status="healthy",
        service="TTS-Cloud (Talk-to-Syllabus)",
        version="1.1.0",
        groq_model=_engine.settings.GROQ_MODEL_NAME,
        embedding_model=_engine.settings.EMBEDDING_MODEL_NAME,
        pinecone_index=_engine.settings.PINECONE_INDEX_NAME,
        total_indexed_chunks=len(_retriever._corpus_cache),
        uptime_seconds=round(time.time() - _start_time, 2)
    )


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def ingest_syllabus_pdf(file: UploadFile = File(...)):
    """
    Ingest, parse, chunk, embed, and index a syllabus PDF.
    Preserves academic Unit/Module hierarchy and extracts breadcrumbs.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF (.pdf) documents are accepted."
        )

    limit = get_max_upload_bytes()
    if file.size is not None and file.size > limit:
        return _upload_too_large(limit)

    try:
        pdf_bytes = await file.read(limit + 1)
        if len(pdf_bytes) > limit:
            return _upload_too_large(limit)
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

        doc_id, chunks = process_pdf_into_chunks(pdf_bytes, filename=file.filename)
        num_indexed = _retriever.index_syllabus_chunks(chunks)

        distinct_units = sorted(list({c.unit for c in chunks if c.unit != "General"}))

        return IngestResponse(
            document_id=doc_id,
            filename=file.filename,
            num_chunks=num_indexed,
            detected_units=distinct_units,
            message=f"Successfully indexed {num_indexed} chunks from '{file.filename}'."
        )
    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning("PDF Ingestion validation error: %s", ve)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Unexpected error during PDF ingestion: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {e}")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval & QA"])
def query_syllabus(req: QueryRequest):
    """
    Execute full RAG query cycle:
    1. Conversational query reformulation.
    2. Hybrid Pinecone dense + BM25 sparse retrieval with RRF.
    3. Grounded answering with precise page and unit citations.
    """
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in req.chat_history]
        resp: RAGResponse = _engine.answer_question(
            question=req.question,
            chat_history=history,
            document_filter=req.document_filter,
            top_k=req.top_k
        )

        citation_items = [
            CitationItem(
                source_id=c["source_id"],
                filename=c["filename"],
                page=c["page"],
                unit=c["unit"],
                section=c["section"],
                score=c["score"],
                snippet=c["snippet"]
            )
            for c in resp.citations
        ]

        return QueryResponse(
            answer=resp.answer,
            standalone_query=resp.standalone_query,
            confidence_score=resp.confidence_score,
            is_low_confidence=resp.is_low_confidence,
            citations=citation_items,
            latency_ms=resp.latency_ms
        )
    except Exception as e:
        logger.error("Query execution failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {e}")


@app.post("/api/v1/study/summary", response_model=UnitSummaryResponse, tags=["Study Utilities"])
def generate_unit_summary(req: UnitSummaryRequest):
    """Generate structured unit/module summary from syllabus context."""
    try:
        summary_text = _engine.generate_unit_summary(
            unit_name_or_query=req.unit_name_or_query,
            document_filter=req.document_filter
        )
        return UnitSummaryResponse(
            unit_query=req.unit_name_or_query,
            summary=summary_text
        )
    except Exception as e:
        logger.error("Unit summary generation failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/study/mcqs", response_model=MCQResponse, tags=["Study Utilities"])
def generate_practice_mcqs(req: MCQRequest):
    """Generate practice exam MCQs grounded in uploaded syllabus."""
    try:
        mcqs_text = _engine.generate_practice_mcqs(
            topic_or_unit=req.topic_or_unit,
            num_questions=req.num_questions,
            document_filter=req.document_filter
        )
        return MCQResponse(
            topic=req.topic_or_unit,
            mcqs=mcqs_text
        )
    except Exception as e:
        logger.error("MCQ generation failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/study/exam-topics", response_model=ExamTopicsResponse, tags=["Study Utilities"])
def extract_exam_topics(req: ExamTopicsRequest):
    """Extract high-weightage exam topics and grading scheme from active syllabus."""
    try:
        topics_text = _engine.extract_key_exam_topics(document_filter=req.document_filter)
        return ExamTopicsResponse(strategy_guide=topics_text)
    except Exception as e:
        logger.error("Exam topics extraction failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete("/api/v1/documents/{document_id}", tags=["Ingestion"])
def delete_document(document_id: str):
    """Delete all indexed chunks for a document from Pinecone and local cache."""
    success = _retriever.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")
    return {"message": f"Document '{document_id}' successfully removed from indices."}
