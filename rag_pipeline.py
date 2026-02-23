"""
rag_pipeline.py - Fixed model + syntax errors
"""
import re
import fitz  # PyMuPDF
import hashlib
import os
from typing import Tuple, List
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq

# ─── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "syllabus-rag")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# ─── Embedding Model ─────────────────────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# ─── Pinecone Setup ─────────────────────────────────────────────────────────
def init_pinecone(index_name: str = PINECONE_INDEX_NAME) -> Pinecone.Index:
    """Auto-initializes Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

# ─── PDF Processing ─────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind('. ')
        if last_period > chunk_size // 2:
            chunk = chunk[:last_period + 1]
        chunks.append(chunk.strip())
        start += len(chunk) - overlap
    return [c for c in chunks if len(c) > 50]

def process_pdf(pdf_bytes: bytes) -> int:
    """PDF → chunks → embeddings → Pinecone (auto-initializes)."""
    index = init_pinecone()
    text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(text)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    
    batch_size = 50
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:16] + f"_{i}"
        vectors.append({
            "id": chunk_id,
            "values": emb,
            "metadata": {"text": chunk, "chunk_index": i}
        })
    
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    return len(chunks)

# ─── Query & Answer ─────────────────────────────────────────────────────────
def retrieve_context(question: str, index, top_k: int = 4) -> Tuple[str, str]:
    embedder = get_embedder()
    q_embedding = embedder.encode([question])[0].tolist()
    results = index.query(vector=q_embedding, top_k=top_k, include_metadata=True)
    chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n---\n\n".join(chunks)
    sources = " | ".join([f"Chunk {i+1}: {c[:60]}..." for i, c in enumerate(chunks)])
    return context, sources

def answer_question(question: str) -> Tuple[str, str]:
    """Complete RAG: retrieve + generate answer."""
    index = init_pinecone()
    context, sources = retrieve_context(question, index)
    
    system_prompt = """You are a helpful academic assistant that answers questions about course syllabi.
Use ONLY the provided context to answer. If the answer is not in the context, say so clearly.
Be concise, accurate, and student-friendly. Format lists and key info clearly."""
    
    user_prompt = f"""Context from the syllabus:
{context}

Student Question: {question}

Answer based on the syllabus context above:"""
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ FIXED: Current working model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )
    answer = response.choices[0].message.content.strip()
    return answer, sources
