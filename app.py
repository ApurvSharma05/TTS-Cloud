import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import process_pdf, answer_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Talk to Syllabus", page_icon="📚", layout="centered")

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0f0e0c; color: #e8e0d0; }
.main-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #f0c060; letter-spacing: -1px; margin-bottom: 0; }
.subtitle { color: #8a7f70; font-size: 1rem; margin-top: 4px; margin-bottom: 2rem; }
.chat-bubble-user { background: #1e1c18; border: 1px solid #3a3530; border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 8px 0; color: #e8e0d0; }
.chat-bubble-ai { background: #1a1f14; border: 1px solid #2e3a28; border-radius: 12px 12px 12px 4px; padding: 12px 16px; margin: 8px 0; color: #c8ddb8; }
.source-tag { font-size: 0.72rem; color: #6a6050; margin-top: 6px; }
.status-pill { display: inline-block; background: #1e2a18; color: #7acc50; border: 1px solid #3a5030; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 Talk to Syllabus</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any course syllabus PDF and ask questions about it</div>', unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ─── PDF Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"])

if uploaded_file:
    if not st.session_state.pdf_processed:
        with st.spinner("🔍 Reading & indexing your syllabus..."):
            try:
                num_chunks = process_pdf(uploaded_file.read())
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.markdown(f'<div class="status-pill">✅ Indexed {num_chunks} chunks from "{uploaded_file.name}"</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()
    else:
        st.markdown(f'<div class="status-pill">✅ Syllabus ready — "{st.session_state.pdf_name}"</div>',
                    unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.info("👈 Upload a syllabus PDF to start asking questions.")

# ─── Chat Interface ─────────────────────────────────────────────────────────
if st.session_state.pdf_processed:
    st.divider()
    st.markdown("### 💬 Ask a Question")

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown(f'<div class="source-tag">📎 Sources: {msg["sources"]}</div>',
                            unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "What are the prerequisites for this course?",
            "What topics are covered in Unit 3?",
            "How is the final grade calculated?",
            "What are the assignment deadlines?"
        ]
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion

    # Handle suggested question
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    # Text input
    question = st.chat_input("Ask anything about your syllabus...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    # Clear chat
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
