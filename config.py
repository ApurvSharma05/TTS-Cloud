"""
config.py - Centralized configuration for Talk-to-Syllabus (TTS-Cloud) RAG System.
Handles environment variables, model parameters, retrieval thresholds, and index settings.
"""

import logging
import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured application logging across the TTS-Cloud system."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("tts-cloud")

logger = setup_logging()

@dataclass(frozen=True)
class RAGSettings:
    """Central configuration parameters for TTS-Cloud."""
    
    # Vector Database (Pinecone)
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "syllabus-rag")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    VECTOR_DIMENSION: int = 384
    VECTOR_METRIC: str = "cosine"
    
    # Embedding Model (Hugging Face)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Generation Model (Groq)
    # Default to qwen/qwen3.8-27b or llama-3.1-8b-instant with automated fallback
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_REFORMULATION_MODEL: str = os.getenv("GROQ_REFORMULATION_MODEL", "qwen/qwen3.8-27b")
    FALLBACK_MODELS: Tuple[str, ...] = (
        "qwen/qwen3.8-27b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b"
    )
    DEFAULT_TEMPERATURE: float = 0.2
    MAX_OUTPUT_TOKENS: int = 768
    
    # Chunking Parameters (Syllabus & Page-Aware)
    CHUNK_SIZE: int = 600  # Target characters per chunk
    CHUNK_OVERLAP: int = 120  # Overlap between consecutive chunks
    MIN_CHUNK_LENGTH: int = 60  # Discard micro chunks/empty noise
    
    # Retrieval & Reranking Settings
    INITIAL_TOP_K: int = 8      # Initial candidate pool retrieved from dense + sparse
    FINAL_TOP_K: int = 4        # Top candidate chunks passed to LLM after reranking/fusion
    RRF_K: int = 60             # Reciprocal Rank Fusion constant
    CONFIDENCE_THRESHOLD: float = 0.35  # Minimum similarity score before triggering low-confidence guardrail
    
    # Conversational History Window
    MAX_HISTORY_TURNS: int = 4   # Last N user/assistant message pairs to consider for reformulation


def get_api_keys() -> Tuple[str, str, str]:
    """
    Retrieve API keys with validation.
    Returns:
        (pinecone_api_key, groq_api_key, pinecone_index_name)
    """
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", RAGSettings.PINECONE_INDEX_NAME).strip()
    
    return pinecone_key, groq_key, index_name
