# 📚 TTS-Cloud (Talk-to-Syllabus)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000.svg)](https://www.pinecone.io/)
[![Groq Llama-3.1](https://img.shields.io/badge/Groq-Llama--3.1--8B-F05A28.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TTS-Cloud (Talk-to-Syllabus)** is a production-grade, domain-specific Retrieval-Augmented Generation (RAG) system engineered specifically for academic course syllabi, university curricula, and textbook modules. 
> 
> Unlike generic "Chat with PDF" wrappers, TTS-Cloud preserves hierarchical curriculum semantics (Units, Modules, Sections, Prerequisites, Marking Schemes), utilizes **Hybrid Dense + Lexical (BM25) Retrieval with Reciprocal Rank Fusion (RRF)**, contextualizes conversational follow-ups, and enforces strict **source citations with exact page numbers and evidence inspection**.

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
    │   • Dense Cosine Similarity (Pinecone Top-K)                  │
    │   • Sparse BM25 Keyword Matching (Exact Acronyms/Codes)       │
    │   • Reciprocal Rank Fusion (RRF k=60) + Blended Scoring       │
    │   • Retrieval Confidence Thresholding (Guardrail)             │
    └───────────────────────────────┬───────────────────────────────┘
                                    │ Top Candidates + Exact Citations
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                 Strictly Grounded RAG Generator               │
    │   • System Guardrails (No hallucinated policies/marks)        │
    │   • Groq Llama-3.1-8B-Instant Inference                       │
    │   • Exact Citation Annotations [Filename, Page, Unit]         │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                     Modern Streamlit Dashboard                │
    │   • Multi-Syllabus Document Library & Selector                │
    │   • Conversational Chat with Expandable Source Evidence       │
    │   • Academic Study Tools (Unit Summary, MCQs, Exam Topics)    │
    │   • Integrated Benchmark Evaluation Suite                     │
    └───────────────────────────────────────────────────────────────┘
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
- **Sparse Lexical Retrieval:** Self-contained Okapi BM25 index over the active syllabus corpus.
- **Reciprocal Rank Fusion (RRF):**
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{w_m}{k + rank_m(d)}$$
  yielding superior candidate ranking over single-modality retrievers.

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

### 6. 📊 Integrated Benchmark Evaluation Suite
Includes an automated evaluation framework (`evaluation.py`) measuring:
- **Hit Rate @ Top-K**
- **Mean Reciprocal Rank (MRR)**
- **Answer Groundedness & Keyword Coverage**
- **Retrieval & Total End-to-End Latency**

---

## 📂 Project Structure

```
TTS-Cloud/
├── config.py                 # Centralized configuration, parameters, and key resolution
├── document_processor.py     # PDF parsing, SHA-256 hashing, unit detection, recursive chunking
├── retriever.py              # Dense Pinecone client, BM25Okapi, RRF rank fusion, candidate reranker
├── rag_engine.py             # Query reformulation, grounded prompt builder, Groq client, academic tools
├── evaluation.py             # Benchmark dataset, Hit@K, MRR, groundedness, and latency evaluator
├── app.py                    # Production Streamlit UI (multi-doc manager, chat, study tools, eval)
├── rag_pipeline.py           # Clean backward-compatible interface
├── requirements.txt          # Production dependencies
├── render.yaml               # Infrastructure-as-code for Render deployment
├── Procfile                  # Procfile for web dynos
├── .env.example              # Environment variables template
└── README.md                 # Complete system documentation
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
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running the Benchmark Evaluation

Run the automated evaluation suite via CLI:
```bash
python evaluation.py
```

Sample Benchmark Output:
```text
=================================================================
🚀 RUNNING TTS-CLOUD RAG BENCHMARK EVALUATION SUITE
=================================================================

📊 Evaluated 5 Benchmark Queries:
  • Hit Rate @ Top-4:       100.0%
  • Mean Reciprocal Rank:   0.883
  • Keyword Coverage:       95.0%
  • Groundedness / Overlap: 82.4%
  • Avg Retrieval Latency:  18.4 ms
  • Avg Total End-to-End:   420.6 ms
=================================================================
```

---

## 🌐 Cloud Deployment

### Option A: Streamlit Community Cloud (Recommended)
1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and link your repository (`app.py`).
3. Under **Advanced Settings $\rightarrow$ Secrets**, add:
   ```toml
   PINECONE_API_KEY = "your_pinecone_api_key"
   GROQ_API_KEY = "your_groq_api_key"
   PINECONE_INDEX_NAME = "syllabus-rag"
   ```
4. Click **Deploy**.

### Option B: Render Web Service
1. Connect your repository to [Render](https://render.com).
2. Render will automatically detect `render.yaml`.
3. Set `PINECONE_API_KEY` and `GROQ_API_KEY` in the Render dashboard environment settings.

---

## 🛡️ Guardrails & Safety
- **Anti-Hallucination:** Answers strictly state when a topic or rule is absent from the syllabus.
- **Confidence Threshold:** Drops low-confidence retrieval matches to prevent noisy prompting.
- **Zero-Storage Secrets:** API keys are never persisted to disk or exposed in the UI.

---

## 📜 License
MIT License. Created by [Apurv Sharma](https://github.com/ApurvSharma05).
