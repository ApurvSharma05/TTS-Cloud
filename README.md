#  Talk-to-Syllabus RAG

A free-tier RAG app: upload a syllabus PDF, ask questions, get AI answers.

## Stack
| Component | Service | Cost |
|-----------|---------|------|
| LLM | Groq (Llama 3 8B) | Free |
| Embeddings | HuggingFace sentence-transformers | Free (local) |
| Vector DB | Pinecone | Free starter |
| Hosting | Streamlit Community Cloud | Free |

## Local Setup

```bash
# 1. Clone / download files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push code to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Deploy — no env vars needed (keys entered via sidebar UI)

## Pinecone Index Setup

1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a new index:
   - **Name**: `syllabus-rag` (or any name)
   - **Dimensions**: `384`
   - **Metric**: `cosine`
3. Copy your API key

## Groq API Key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key (free, generous rate limits)

## How It Works

```
PDF Upload
    ↓
Extract text (PyMuPDF)
    ↓
Split into 500-char chunks with 100-char overlap
    ↓
Embed with all-MiniLM-L6-v2 (384-dim vectors)
    ↓
Store in Pinecone
    ↓
User Question → Embed → Query Pinecone (top 4 chunks)
    ↓
Send context + question to Groq (Llama 3 8B)
    ↓
Display Answer
```
