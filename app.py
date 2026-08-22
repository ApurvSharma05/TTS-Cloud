"""
app.py - Production Streamlit Application for Talk-to-Syllabus (TTS-Cloud).
Features Multi-Document Management, Academic Syllabus Study Tools,
Expandable Evidence Citations, Conversational Query Memory, and RAG Benchmark Evaluation.
"""

import os
import streamlit as st
from typing import Dict, List, Optional

from config import RAGSettings, get_api_keys
from document_processor import process_pdf_into_chunks
from retriever import SyllabusRetriever, RetrievedDocument
from rag_engine import SyllabusRAGEngine, RAGResponse
from evaluation import RAGEvaluator, BENCHMARK_EVAL_DATASET

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS-Cloud | Talk to Syllabus",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS & High-Aesthetic Theme ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background & Core Theme */
.stApp {
    background-color: #0b0f14;
    color: #e2e8f0;
}

/* App Header styling */
.hero-container {
    background: linear-gradient(135deg, #131b26 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.02rem;
    font-weight: 400;
}

/* Badges & Tags */
.badge-tech {
    display: inline-block;
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #0284c7;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}

.badge-unit {
    display: inline-block;
    background: #312e81;
    color: #c7d2fe;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.74rem;
    font-weight: 600;
}

.badge-page {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.74rem;
    font-weight: 600;
    margin-left: 6px;
}

.badge-score {
    display: inline-block;
    background: #451a03;
    color: #fdba74;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.74rem;
    font-weight: 600;
    margin-left: 6px;
}

/* Chat Messages */
.chat-user-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #f8fafc;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.chat-assistant-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 4px 14px 14px 14px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #e2e8f0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    line-height: 1.6;
}

.reformulated-query-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: #111827;
    border: 1px dashed #4b5563;
    border-radius: 6px;
    padding: 6px 12px;
    color: #a5b4fc;
    margin-bottom: 12px;
}

/* Evidence Card */
.evidence-card {
    background: #090d13;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    margin-bottom: 12px;
}

.evidence-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #e2e8f0;
}

.evidence-snippet {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    color: #94a3b8;
    background: #020617;
    padding: 8px 12px;
    border-radius: 6px;
    border-left: 3px solid #6366f1;
    white-space: pre-wrap;
}

/* Metric Box */
.metric-card {
    background: #131b26;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tab Active Highlights */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    background-color: #111827;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #0284c7 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {}
if "selected_doc_filter" not in st.session_state:
    st.session_state.selected_doc_filter = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ─── Initialize Shared RAG Pipeline & Retriever ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_engine(pinecone_key: str, groq_key: str, index_name: str) -> SyllabusRAGEngine:
    """Cached initialization of the heavy ML models and API clients."""
    retriever = SyllabusRetriever(pinecone_api_key=pinecone_key, index_name=index_name)
    engine = SyllabusRAGEngine(retriever=retriever, groq_api_key=groq_key)
    return engine

# ─── Sidebar: API Keys & Multi-Document Manager ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Environment keys with UI fallback
    default_p_key, default_g_key, default_idx = get_api_keys()
    
    with st.expander("🔑 API Credentials & Index", expanded=not (default_p_key and default_g_key)):
        pinecone_api_key = st.text_input(
            "Pinecone API Key",
            value=default_p_key,
            type="password",
            help="Get your free key at console.pinecone.io"
        )
        groq_api_key = st.text_input(
            "Groq API Key",
            value=default_g_key,
            type="password",
            help="Get your free key at console.groq.com"
        )
        index_name = st.text_input("Pinecone Index Name", value=default_idx)

    # Validate keys
    keys_configured = bool(pinecone_api_key and groq_api_key)
    if keys_configured:
        st.caption("🟢 **Status:** API Keys Active")
    else:
        st.warning("⚠️ Enter Pinecone & Groq keys to enable RAG indexing and generation.")

    st.divider()

    # RAG Settings & Architecture Specs
    with st.expander("🛠️ RAG Pipeline Specs", expanded=False):
        st.markdown(f"""
        - **Embedding Model:** `all-MiniLM-L6-v2` (384-dim)
        - **LLM Generator:** `llama-3.1-8b-instant`
        - **Retrieval:** Hybrid (Dense Cosine + Sparse BM25)
        - **Fusion:** Reciprocal Rank Fusion (RRF $k=60$)
        - **Chunk Size:** 600 chars (Unit & Page Aware)
        - **Candidate Pool:** Top 8 $\\rightarrow$ Rerank Top 4
        """)

    st.divider()

    # ─── Document Management Section ──────────────────────────────────────────
    st.markdown("### 📁 Syllabus Library")
    
    # PDF Ingestion Uploader
    uploaded_files = st.file_uploader(
        "Upload Syllabus PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload course syllabi (e.g. DBMS.pdf, Operating-Systems.pdf)"
    )

    if uploaded_files and keys_configured:
        try:
            rag_engine = load_rag_engine(pinecone_api_key, groq_api_key, index_name)
            for file_obj in uploaded_files:
                if file_obj.name not in st.session_state.indexed_docs:
                    with st.spinner(f"⚡ Ingesting & indexing '{file_obj.name}'..."):
                        pdf_bytes = file_obj.read()
                        doc_id, chunks = process_pdf_into_chunks(pdf_bytes, filename=file_obj.name)
                        
                        progress_bar = st.progress(0.0, text="Upserting embeddings to Pinecone...")
                        num_upserted = rag_engine.retriever.index_syllabus_chunks(
                            chunks,
                            progress_callback=lambda p: progress_bar.progress(p)
                        )
                        progress_bar.empty()

                        # Collect distinct units
                        distinct_units = sorted(list({c.unit for c in chunks if c.unit != "General"}))
                        distinct_pages = max(c.page for c in chunks) if chunks else 1

                        st.session_state.indexed_docs[file_obj.name] = {
                            "document_id": doc_id,
                            "filename": file_obj.name,
                            "chunk_count": num_upserted,
                            "total_pages": distinct_pages,
                            "units": distinct_units
                        }
                        st.toast(f"✅ Successfully indexed {num_upserted} chunks from {file_obj.name}", icon="📚")
        except Exception as e:
            st.error(f"Error during PDF processing: {e}")

    # Display Indexed Syllabi List
    if st.session_state.indexed_docs:
        st.markdown(f"**Indexed Documents ({len(st.session_state.indexed_docs)}):**")
        doc_names = list(st.session_state.indexed_docs.keys())
        
        for name in doc_names:
            doc_info = st.session_state.indexed_docs[name]
            with st.container():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"📄 **{name}**\n\n<span style='font-size:0.75rem; color:#94a3b8;'>{doc_info['total_pages']} pages • {doc_info['chunk_count']} chunks</span>", unsafe_allow_html=True)
                if c2.button("🗑️", key=f"del_{doc_info['document_id']}", help=f"Remove {name}"):
                    if keys_configured:
                        rag_engine = load_rag_engine(pinecone_api_key, groq_api_key, index_name)
                        rag_engine.retriever.delete_document(doc_info['document_id'])
                    del st.session_state.indexed_docs[name]
                    st.rerun()

        # Multi-document filter dropdown
        filter_options = ["🌐 All Indexed Syllabi"] + doc_names
        selected_filter = st.selectbox("Active Search Scope:", filter_options)
        if selected_filter == "🌐 All Indexed Syllabi":
            st.session_state.selected_doc_filter = None
        else:
            st.session_state.selected_doc_filter = st.session_state.indexed_docs[selected_filter]["document_id"]
    else:
        st.info("No syllabus uploaded yet. Upload a PDF above to start querying.")

# ─── Main Interface Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📚 Talk to Syllabus (TTS-Cloud)</div>
    <div class="hero-subtitle">
        Intelligent Academic RAG System with Unit-Aware Chunking, Hybrid BM25+Pinecone Retrieval, Exact Citations & Guardrails.
    </div>
    <div style="margin-top: 14px;">
        <span class="badge-tech">⚡ Groq Llama-3.1</span>
        <span class="badge-tech">🌲 Pinecone Vector DB</span>
        <span class="badge-tech">🔍 Hybrid BM25 + RRF</span>
        <span class="badge-tech">🎯 Unit & Page Aware</span>
        <span class="badge-tech">🛡️ Strict Guardrails</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab_chat, tab_study, tab_eval = st.tabs([
    "💬 Academic Chat Assistant",
    "🎓 Syllabus Study Tools",
    "📊 RAG Diagnostics & Evaluation"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ACADEMIC CHAT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not keys_configured:
        st.warning("⚠️ Please provide your Pinecone and Groq API keys in the sidebar to begin.")
    elif not st.session_state.indexed_docs:
        st.info("👈 Upload your course syllabus PDF in the sidebar to activate the assistant.")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user-box">🧑‍🎓 <b>Student:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant-box">🤖 <b>TTS Assistant:</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
                
                # Show conversational reformulation badge if applicable
                if msg.get("standalone_query") and msg["standalone_query"] != msg.get("original_question"):
                    st.markdown(
                        f'<div class="reformulated-query-box">🔍 <b>Contextualized Search Query:</b> {msg["standalone_query"]}</div>',
                        unsafe_allow_html=True
                    )

                # Expandable Verified Sources & Evidence Chunks
                if msg.get("citations"):
                    with st.expander(f"📚 Verified Sources & Evidence Chunks ({len(msg['citations'])})", expanded=False):
                        for cit in msg["citations"]:
                            st.markdown(f"""
                            <div class="evidence-card">
                                <div class="evidence-header">
                                    <span>📄 {cit['filename']}</span>
                                    <div>
                                        <span class="badge-unit">{cit['unit']}</span>
                                        <span class="badge-page">Page {cit['page']}</span>
                                        <span class="badge-score">Score: {cit['score']:.3f}</span>
                                    </div>
                                </div>
                                <div style="font-size:0.75rem; color:#64748b; margin-bottom:4px;">Section: {cit['section']}</div>
                                <div class="evidence-snippet">{cit['snippet']}</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Suggested Questions for Quick Start
        if not st.session_state.messages:
            st.markdown("**💡 Quick Start Questions:**")
            col1, col2 = st.columns(2)
            sample_prompts = [
                "What are the prerequisites and learning objectives for this course?",
                "List all topics and subtopics covered in Unit 3.",
                "What is the evaluation and grading scheme for internal exams?",
                "Which reference and text books are recommended in the syllabus?"
            ]
            for i, p in enumerate(sample_prompts):
                target_col = col1 if i % 2 == 0 else col2
                if target_col.button(f"📌 {p}", key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_question = p

        # Query Handler Function
        def handle_user_query(user_query: str):
            rag_engine = load_rag_engine(pinecone_api_key, groq_api_key, index_name)
            
            # Format recent history for query reformulation
            history_context = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            with st.spinner("🧠 Analyzing syllabus context with Hybrid RAG..."):
                resp: RAGResponse = rag_engine.answer_question(
                    question=user_query,
                    chat_history=history_context,
                    document_filter=st.session_state.selected_doc_filter,
                    top_k=4
                )

            # Append user and assistant turn
            st.session_state.messages.append({
                "role": "user",
                "content": user_query
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "citations": resp.citations,
                "standalone_query": resp.standalone_query,
                "original_question": user_query,
                "confidence_score": resp.confidence_score,
                "latency_ms": resp.latency_ms
            })

        # Process pending question if triggered by button
        if st.session_state.pending_question:
            q = st.session_state.pop("pending_question")
            handle_user_query(q)
            st.rerun()

        # Chat Input Box
        user_input = st.chat_input("Ask any question about your syllabus (e.g. 'Explain 3NF from Unit 3', 'What are the course outcomes?')...")
        if user_input:
            handle_user_query(user_input)
            st.rerun()

        # Controls row
        if st.session_state.messages:
            c1, c2, c3 = st.columns([1, 1, 4])
            if c1.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ACADEMIC SYLLABUS STUDY TOOLS
# ══════════════════════════════════════════════════════════════════════════════
with tab_study:
    st.markdown("### 🎓 Academic Syllabus Study Utilities")
    st.caption("Generate exam prep materials, unit summaries, and practice MCQs directly grounded in your uploaded syllabus.")

    if not keys_configured or not st.session_state.indexed_docs:
        st.info("Please configure API keys and upload at least one syllabus PDF to use these study tools.")
    else:
        rag_engine = load_rag_engine(pinecone_api_key, groq_api_key, index_name)
        
        tool_choice = st.radio(
            "Select Study Feature:",
            ["📋 Comprehensive Unit Summary", "❓ Practice MCQ Generator", "🌟 Key Exam Topics & Strategy"],
            horizontal=True
        )

        st.divider()

        if tool_choice == "📋 Comprehensive Unit Summary":
            st.markdown("#### 📋 Generate Unit / Module Summary")
            u_input = st.text_input("Enter Unit Name or Number (e.g., 'Unit 3: Normalization' or 'Module 2: Process Management'):")
            if st.button("Generate Unit Summary", type="primary"):
                if u_input.strip():
                    with st.spinner("Generating structured syllabus summary..."):
                        summary_text = rag_engine.generate_unit_summary(
                            unit_name_or_query=u_input,
                            document_filter=st.session_state.selected_doc_filter
                        )
                        st.markdown(summary_text)
                else:
                    st.warning("Please enter a unit name or topic.")

        elif tool_choice == "❓ Practice MCQ Generator":
            st.markdown("#### ❓ Generate University Practice MCQs")
            col_t1, col_t2 = st.columns([3, 1])
            topic_input = col_t1.text_input("Topic or Unit for MCQs:", value="Unit 2")
            mcq_count = col_t2.slider("Number of Questions:", min_value=2, max_value=6, value=4)
            
            if st.button("Generate Practice MCQs", type="primary"):
                if topic_input.strip():
                    with st.spinner("Crafting syllabus-grounded exam MCQs..."):
                        mcq_output = rag_engine.generate_practice_mcqs(
                            topic_or_unit=topic_input,
                            num_questions=mcq_count,
                            document_filter=st.session_state.selected_doc_filter
                        )
                        st.markdown(mcq_output)
                else:
                    st.warning("Please specify a topic or unit.")

        elif tool_choice == "🌟 Key Exam Topics & Strategy":
            st.markdown("#### 🌟 Key Exam Topics & Syllabus Breakdown")
            st.write("Extracts high-yield topics, textbook references, and grading structure from the active syllabus.")
            if st.button("Analyze Syllabus & Extract Key Topics", type="primary"):
                with st.spinner("Analyzing syllabus structure..."):
                    topics_output = rag_engine.extract_key_exam_topics(
                        document_filter=st.session_state.selected_doc_filter
                    )
                    st.markdown(topics_output)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RAG DIAGNOSTICS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 RAG Benchmark Evaluation Suite")
    st.markdown(
        "Evaluate retrieval precision, ranking quality, keyword coverage, and latency "
        "across standard academic syllabus test queries."
    )

    if st.button("▶️ Run RAG Benchmark Evaluation", type="primary"):
        if not keys_configured:
            st.warning("Please configure API keys in the sidebar first.")
        else:
            with st.spinner("Running automated benchmark test suite..."):
                rag_engine = load_rag_engine(pinecone_api_key, groq_api_key, index_name)
                evaluator = RAGEvaluator(engine=rag_engine)
                benchmark_summary = evaluator.run_benchmark(top_k=4)

                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{benchmark_summary.mean_hit_at_k * 100:.1f}%</div>
                    <div class="metric-label">Hit Rate @ Top-4</div>
                </div>
                """, unsafe_allow_html=True)

                m2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{benchmark_summary.mean_mrr:.3f}</div>
                    <div class="metric-label">Mean Reciprocal Rank</div>
                </div>
                """, unsafe_allow_html=True)

                m3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{benchmark_summary.mean_keyword_coverage * 100:.1f}%</div>
                    <div class="metric-label">Keyword Coverage</div>
                </div>
                """, unsafe_allow_html=True)

                m4.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{benchmark_summary.avg_total_latency_ms:.0f} ms</div>
                    <div class="metric-label">Avg E2E Latency</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📋 Test Case Evaluation Breakdown")
                st.dataframe(
                    benchmark_summary.detailed_results,
                    column_config={
                        "test_id": "Test ID",
                        "question": "Query",
                        "hit_at_k": "Hit @ Top-4",
                        "reciprocal_rank": "MRR",
                        "keyword_coverage": "KW Match",
                        "groundedness_score": "Groundedness",
                        "retrieval_latency_ms": "Retrieval (ms)",
                        "generation_latency_ms": "Gen (ms)",
                        "confidence_score": "Confidence"
                    },
                    use_container_width=True
                )

    st.divider()
    with st.expander("📖 Inspect Benchmark Evaluation Dataset Details", expanded=False):
        st.json(BENCHMARK_EVAL_DATASET)
