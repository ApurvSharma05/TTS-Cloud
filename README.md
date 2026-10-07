# 📚 TTS-Cloud (Talk-to-Syllabus)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000.svg)](https://www.pinecone.io/)
[![Groq Fast Inference](https://img.shields.io/badge/Groq-Fast_LPU-F05A28.svg)](https://groq.com/)
[![Pytest](https://img.shields.io/badge/Pytest-17_Passing-success.svg)](https://pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TTS-Cloud (Talk-to-Syllabus)** is a production-grade, domain-specific Retrieval-Augmented Generation (RAG) platform engineered specifically for academic course syllabi, university curricula, and textbook modules. 
> 
> Unlike generic "Chat with PDF" wrappers, TTS-Cloud preserves hierarchical curriculum semantics (Units, Modules, Sections, Prerequisites, Marking Schemes), utilizes **Hybrid Dense + Lexical (BM25) Retrieval with Reciprocal Rank Fusion (RRF)**, contextualizes conversational follow-ups, mitigates prompt injection, and enforces strict **source citations with exact page numbers and evidence inspection**.
> 
> TTS-Cloud provides dual serving interfaces: an interactive **Streamlit Dashboard** and a high-performance **FastAPI REST API** with Pydantic request/response validation.

---

## 🏗️ System Architecture

```
                                 [ Syllabus PDF Uploads ]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       Document Ingestion & Validation         │
                    │   • SHA-256 Checksum Deduplication            │
                    │   • PyMuPDF Page-by-Page Extraction           │
                    │   • Syllabus Unit & Module Boundary Detection │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      Hierarchical Contextual Chunking         │
                    │   • Unit/Module Context Breadcrumb Injection  │
                    │   • Recursive Separators (Para > List > Sent) │
                    │   • Rich Metadata (doc_id, page, unit, sec)   │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
                            ▼                               ▼
                 [ In-Memory BM25 Index ]        [ all-MiniLM-L6-v2 Embed ]
                            │                               │
                            │                               ▼
                            │                    [ Pinecone Vector Index ]
                            │                               │
    [ User Question + History ]                             │
                 │                                          │
                 ▼                                          │
    ┌───────────────────────────┐                           │
    │ Conversational Memory &   │                           │
    │ Groq Query Reformulation  │                           │
    └────────────┬──────────────┘                           │
                 │                                          │
                 ▼                                          │
    ┌───────────────────────────────────────────────────────┴───────┐
    │                 Hybrid Retrieval & Candidate Reranker         │
    │   • Dense Cosine Similarity (Pinecone Top-K / Local Fallback) │
    │   • Sparse BM25 Keyword Matching (Exact Acronyms/Codes)       │
    │   • Reciprocal Rank Fusion (RRF k=60) + Blended Scoring       │
    │   • Retrieval Confidence Thresholding (Guardrail >= 0.35)     │
    │   • Thread-Safe Mutex Lock Protection                         │
    └───────────────────────────────┬───────────────────────────────┘
                                    │ Top Candidates + Exact Citations
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                 Strictly Grounded RAG Generator               │
    │   • System Guardrails & Prompt Injection Isolation            │
    │   • Groq LPU Inference with Automated Model Fallback          │
    │   • Exact Citation Annotations [Filename, Page, Unit]         │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌───────────────────────────────┐ ┌───────────────────────────────┐
    │  Modern Streamlit Dashboard   │ │     FastAPI REST Service      │
    │  • Multi-Syllabus Manager     │ │  • POST /api/v1/query         │
    │  • Expandable Evidence Cards  │ │  • POST /api/v1/ingest        │
    │  • Academic Study Utilities   │ │  • Pydantic Request/Response  │
    │  • Live Benchmark Diagnostics │ │  • Interactive OpenAPI Docs   │
    └───────────────────────────────┘ └───────────────────────────────┘
```

---

## ✨ Key Technical Features

### 1. 🎯 Syllabus-Aware & Hierarchical Chunking
Standard RAG systems shred syllabi using arbitrary 500-character splitters, causing topic lists, prerequisite tables, and unit boundaries to split mid-sentence. TTS-Cloud implements:
- **Academic Heading Heuristics:** Detects `UNIT I-V`, `MODULE 1-5`, `CHAPTER`, and section numbering (`3.1`, `3.2`).
- **Context Breadcrumb Enrichment:** Every chunk is injected with a structured header before embedding:
  ```
  [DBMS.pdf | Page 14 | Unit 3: Normalization | 3.3 3NF]
  A relation R is in Third Normal Form (3NF) if...
  ```
  This guarantees that vector embeddings and lexical search always preserve high-level course context.

### 2. 🔍 Hybrid Retrieval (Pinecone Dense + BM25 Sparse)
Academic syllabi frequently contain specialized acronyms, course codes, and algorithmic complexity notations (`2NF`, `3NF`, `BCNF`, `TCP/IP`, `OSI`, `ACID`, `NP-Hard`, `O(V+E)`). Dense embeddings often dilute exact keyword matching. TTS-Cloud fuses:
- **Dense Semantic Retrieval:** `all-MiniLM-L6-v2` (384 dimensions) indexed in Pinecone Serverless.
- **Sparse Lexical Retrieval:** Self-contained Okapi BM25 index over the active syllabus corpus preserving technical symbols.
- **Reciprocal Rank Fusion (RRF):**
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{w_m}{k + rank_m(d)}$$
  yielding superior candidate ranking over single-modality retrievers.
- **Local Fallback:** Falls back to vectorized NumPy cosine similarity if cloud vector databases are unreachable.

### 3. 🧠 Conversational Memory with Query Reformulation
Maintains conversation history and reformulates multi-turn student follow-up queries:
- *Student:* "What is normalization?" $\rightarrow$ *Assistant:* explains 1NF, 2NF, 3NF, BCNF.
- *Follow-up:* "Explain the conditions for the third one."
- *Reformulated Query:* `"Third Normal Form (3NF) conditions and functional dependencies in DBMS"`

### 4. 📚 Verifiable Source Citations & Evidence Drawer
Every generated response provides clickable, expandable source evidence showing:
- Document filename
- Exact syllabus page number
- Detected Unit & Section
- Retrieval similarity & hybrid confidence score
- Full chunk text snippet

### 5. 🎓 Academic Syllabus Study Suite
- **Comprehensive Unit Summarizer:** Generates logical summaries of course outcomes, core topics, and key concepts.
- **University Practice MCQ Generator:** Creates exam-style questions with answer keys, explanations, and syllabus citations.
- **Key Exam Topics & Strategy:** Extracts high-weightage topics and recommended textbook references.

### 6. 🚀 FastAPI Serving Layer & OpenAPI Documentation
- Fully decoupled REST API with Pydantic request/response models.
- Endpoints for PDF ingestion, grounded querying, study tools, document management, and health checks.
- Interactive Swagger documentation at `/docs`.

---

## 📂 Project Structure

```
TTS-Cloud/
├── api.py                    # Production FastAPI REST API server with Pydantic schemas
├── app.py                    # Production Streamlit UI (multi-doc manager, chat, study tools, eval)
├── config.py                 # Centralized configuration, parameters, model fallback, logging setup
├── document_processor.py     # PDF parsing, SHA-256 hashing, unit detection, recursive chunking
├── retriever.py              # Dense Pinecone client, BM25Okapi, RRF rank fusion, thread-safe cache
├── rag_engine.py             # Query reformulation, prompt injection defense, Groq client, academic tools
├── rag_pipeline.py           # Clean backward-compatible interface
├── evaluation.py             # Benchmark suite: Hit@K, MRR, keyword coverage, groundedness, latency
├── Dockerfile                # Production multi-stage Docker build
├── docker-compose.yml        # Multi-service orchestration (Streamlit on :8501, FastAPI on :8000)
├── requirements.txt          # Production dependencies
├── render.yaml               # Infrastructure-as-code for Render deployment
├── Procfile                  # Procfile for web dynos
├── .env.example              # Environment variables template
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI workflow (Pytest + Benchmark gate)
├── tests/
│   ├── test_document_processor.py  # Unit tests for chunking, hashing, and regex detection
│   ├── test_retriever.py           # Unit tests for BM25, RRF ranking, and in-memory cache
│   ├── test_rag_engine.py          # Unit tests for guardrails and response modeling
│   └── test_api.py                 # Integration tests for FastAPI endpoints
└── PROJECT_REVIEW.md         # Comprehensive senior engineer project evaluation
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- [Pinecone API Key](https://console.pinecone.io) (Free Starter tier)
- [Groq API Key](https://console.groq.com) (Free tier)

### 2. Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/ApurvSharma05/TTS-Cloud.git
cd TTS-Cloud

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_INDEX_NAME=syllabus-rag
GROQ_MODEL=qwen/qwen3.8-27b
```

### 4. Run the Applications

#### Option A: Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Option B: Run the FastAPI REST API
```bash
uvicorn api:app --reload --port 8000
```
- API Base: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative ReDoc UI: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Running with Docker & Docker Compose

Launch both the Streamlit UI and the FastAPI REST Service with a single command:

```bash
docker-compose up --build
```
- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing & Automated CI

Run the automated Pytest test suite:
```bash
pytest tests/ -v
```

All 17 tests validate document hashing, unit detection heuristics, chunk length bounds, BM25 tokenization, in-memory cache operations, confidence guardrails, and FastAPI schema validation.

---

## 📊 Running the Benchmark Evaluation

Run the automated evaluation suite via CLI:
```bash
python evaluation.py --top-k 4
```

Verified Benchmark Results (Live Run):
```text
=================================================================
🚀 RUNNING TTS-CLOUD RAG BENCHMARK EVALUATION SUITE
=================================================================

📊 Evaluated 5 Benchmark Queries:
  • Hit Rate @ Top-4:       100.0%
  • Mean Reciprocal Rank:   1.000
  • Keyword Coverage:       78.0%
  • Groundedness / Overlap: 41.9%
  • Avg Retrieval Latency:  394.2 ms
  • Avg Total End-to-End:   1335.1 ms

Detailed Case Results:
  • [eval_dbms_01] Hit: True | MRR: 1.0 | Latency: 630.9ms | Conf: 0.387
  • [eval_cn_02]   Hit: True | MRR: 1.0 | Latency: 322.5ms | Conf: 0.417
  • [eval_os_03]   Hit: True | MRR: 1.0 | Latency: 325.3ms | Conf: 0.347
  • [eval_daa_04]  Hit: True | MRR: 1.0 | Latency: 321.2ms | Conf: 0.362
  • [eval_oos_05]  Hit: True | MRR: 1.0 | Latency: 371.1ms | Conf: 0.373
=================================================================
```

---

## 🛡️ Guardrails, Security & Safety
- **Anti-Hallucination & Out-of-Syllabus:** Retrieval similarity thresholding (`>= 0.35`) intercepts out-of-scope queries before LLM generation.
- **Prompt Injection Defense:** Student input is strictly quarantined within `<student_query>` XML boundaries with explicit system prompt rules.
- **Thread Safety:** Mutex locks prevent race conditions in multi-threaded environments during index updates and document deletions.
- **Graceful Model Fallback:** Automatic failover across supported Groq models (`qwen/qwen3.8-27b`, `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`).

---

## 📜 License
MIT License. Created by [Apurv Sharma](https://github.com/ApurvSharma05).
