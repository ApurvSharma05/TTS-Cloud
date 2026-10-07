"""
retriever.py - Hybrid Retrieval Engine with Dense Pinecone Vector Search,
BM25 Lexical Keyword Matching, Reciprocal Rank Fusion (RRF), and Candidate Reranking.
"""

import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config import RAGSettings, get_api_keys
from document_processor import SyllabusChunk

logger = logging.getLogger("tts-cloud.retriever")


@dataclass
class RetrievedDocument:
    """Represents a retrieved and reranked context chunk with source metadata."""
    chunk_id: str
    document_id: str
    filename: str
    page: int
    unit: str
    section: str
    text: str
    score: float
    dense_score: float
    sparse_score: float
    rrf_score: float

    def to_citation_header(self) -> str:
        """Formatted citation breadcrumb for prompt and UI."""
        return f"{self.filename} (Page {self.page} — {self.unit})"


class BM25Okapi:
    """
    Lightweight, self-contained Okapi BM25 implementation for lexical keyword matching.
    Provides robust scoring for exact academic keywords, codes, and acronyms (e.g. 2NF, BCNF, ACID).
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs: List[Dict[str, int]] = [Counter(doc) for doc in self.tokenized_corpus]
        self.idf: Dict[str, float] = self._calc_idf()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words preserving hyphens and dots."""
        return re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text.lower())

    def _calc_idf(self) -> Dict[str, float]:
        df: Dict[str, int] = Counter()
        for doc_freq in self.doc_freqs:
            for term in doc_freq:
                df[term] += 1

        idf: Dict[str, float] = {}
        for term, freq in df.items():
            # BM25 IDF formula with smoothing
            idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
        return idf

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for a query across all documents."""
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(token, 0)
                if freq > 0:
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += token_idf * (numerator / denominator)

        return scores


class SyllabusRetriever:
    """
    Production-grade hybrid retriever combining Dense Pinecone embeddings
    and Sparse BM25 keyword matching with Reciprocal Rank Fusion (RRF).
    """

    _embedder_instance = None

    def __init__(self, pinecone_api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.settings = RAGSettings()
        key, _, default_idx = get_api_keys()
        self.api_key = pinecone_api_key or key
        self.index_name = index_name or default_idx
        self._pinecone_client = None
        self._index = None
        self._lock = threading.Lock()
        self._corpus_cache: List[Dict[str, Any]] = []
        self._corpus_embeddings: Optional[np.ndarray] = None
        self._bm25_model: Optional[BM25Okapi] = None

    @classmethod
    def get_embedder(cls) -> SentenceTransformer:
        """Lazy singleton loader for HuggingFace SentenceTransformer."""
        if cls._embedder_instance is None:
            logger.info("Loading SentenceTransformer model: %s", RAGSettings.EMBEDDING_MODEL_NAME)
            cls._embedder_instance = SentenceTransformer(RAGSettings.EMBEDDING_MODEL_NAME)
        return cls._embedder_instance

    def _get_pinecone_index(self):
        """Initialize or retrieve existing Pinecone Index."""
        if self._index is not None:
            return self._index

        if not self.api_key:
            raise ValueError(
                "❌ Pinecone API Key is missing. Please set PINECONE_API_KEY in .env "
                "or pass it in the application sidebar."
            )

        self._pinecone_client = Pinecone(api_key=self.api_key)
        existing_indexes = [idx.name for idx in self._pinecone_client.list_indexes()]

        if self.index_name not in existing_indexes:
            self._pinecone_client.create_index(
                name=self.index_name,
                dimension=self.settings.VECTOR_DIMENSION,
                metric=self.settings.VECTOR_METRIC,
                spec=ServerlessSpec(cloud=self.settings.PINECONE_CLOUD, region=self.settings.PINECONE_REGION)
            )

        self._index = self._pinecone_client.Index(self.index_name)
        return self._index

    def index_syllabus_chunks(
        self,
        chunks: List[SyllabusChunk],
        batch_size: int = 50,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Embed and upsert chunks into Pinecone vector index.
        Also updates the local sparse BM25 and vector cache.
        """
        if not chunks:
            return 0

        embedder = self.get_embedder()
        texts_to_embed = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(texts_to_embed, show_progress_bar=False)

        # Upsert to Pinecone if index is available
        try:
            index = self._get_pinecone_index()
            vectors = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings.tolist())):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": emb,
                    "metadata": chunk.to_metadata()
                })

            total_batches = (len(vectors) + batch_size - 1) // batch_size
            for b_idx in range(0, len(vectors), batch_size):
                batch = vectors[b_idx:b_idx + batch_size]
                index.upsert(vectors=batch)
                if progress_callback:
                    progress_callback(min(1.0, (b_idx // batch_size + 1) / total_batches))
            logger.info("Successfully indexed %d chunks into Pinecone index '%s'.", len(chunks), self.index_name)
        except Exception as e:
            logger.warning("Pinecone upsert skipped or offline (%s). Retaining chunks in local memory cache.", e)

        # Thread-safe update of in-memory corpus & embeddings cache
        with self._lock:
            for chunk in chunks:
                self._corpus_cache.append(chunk.to_metadata())
            
            if self._corpus_embeddings is None:
                self._corpus_embeddings = embeddings
            else:
                self._corpus_embeddings = np.vstack([self._corpus_embeddings, embeddings])

            self._rebuild_bm25()

        return len(chunks)

    def _rebuild_bm25(self):
        """Reconstruct the in-memory BM25 index from active corpus."""
        if not self._corpus_cache:
            self._bm25_model = None
            return
        corpus_texts = [doc.get("text", "") for doc in self._corpus_cache]
        self._bm25_model = BM25Okapi(corpus_texts)

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors for a specific document_id from Pinecone and local cache."""
        try:
            if self.api_key:
                index = self._get_pinecone_index()
                index.delete(filter={"document_id": {"$eq": document_id}})
                logger.info("Deleted document '%s' vectors from Pinecone.", document_id)
        except Exception as e:
            logger.warning("Pinecone deletion skipped or failed: %s", e)

        # Thread-safe removal from local cache
        with self._lock:
            remaining_cache = []
            indices_to_keep = []
            for i, c in enumerate(self._corpus_cache):
                if c.get("document_id") != document_id:
                    remaining_cache.append(c)
                    indices_to_keep.append(i)

            self._corpus_cache = remaining_cache
            if self._corpus_embeddings is not None and len(indices_to_keep) > 0:
                self._corpus_embeddings = self._corpus_embeddings[indices_to_keep]
            else:
                self._corpus_embeddings = None

            self._rebuild_bm25()

        return True

    def hybrid_search(
        self,
        query: str,
        document_filter: Optional[str] = None,
        top_k: int = 4,
        candidate_pool_size: int = 8
    ) -> Tuple[List[RetrievedDocument], float]:
        """
        Execute Hybrid Retrieval:
        1. Dense semantic search via Pinecone (or local vector cosine fallback).
        2. Sparse lexical search via BM25Okapi.
        3. Reciprocal Rank Fusion (RRF) & Relevance blending.
        4. Guardrail filtering & confidence score calculation.

        Returns:
            (list_of_reranked_docs, overall_confidence_score)
        """
        embedder = self.get_embedder()
        q_emb = embedder.encode([query])[0]
        q_emb_list = q_emb.tolist()

        dense_candidates: Dict[str, Dict[str, Any]] = {}

        # Attempt Dense Retrieval from Pinecone
        pinecone_success = False
        if self.api_key:
            try:
                pinecone_filter = {"document_id": {"$eq": document_filter}} if document_filter else None
                index = self._get_pinecone_index()
                dense_results = index.query(
                    vector=q_emb_list,
                    top_k=candidate_pool_size,
                    include_metadata=True,
                    filter=pinecone_filter
                )
                for rank, match in enumerate(dense_results.get("matches", [])):
                    cid = match.get("id")
                    meta = match.get("metadata", {})
                    score = float(match.get("score", 0.0))
                    dense_candidates[cid] = {
                        "rank": rank + 1,
                        "score": score,
                        "metadata": meta
                    }
                pinecone_success = True
            except Exception:
                pinecone_success = False

        # Fallback to local cosine similarity if Pinecone returned nothing or offline
        if not pinecone_success and self._corpus_cache and self._corpus_embeddings is not None:
            # Cosine similarity calculation: (A . B) / (||A|| * ||B||)
            norm_q = np.linalg.norm(q_emb)
            if norm_q > 0:
                norms_docs = np.linalg.norm(self._corpus_embeddings, axis=1)
                norms_docs[norms_docs == 0] = 1e-10
                sims = np.dot(self._corpus_embeddings, q_emb) / (norms_docs * norm_q)
                
                ranked_indices = np.argsort(sims)[::-1]
                for rank, idx in enumerate(ranked_indices[:candidate_pool_size]):
                    meta = self._corpus_cache[idx]
                    if document_filter and meta.get("document_id") != document_filter:
                        continue
                    cid = meta.get("chunk_id", f"chunk_{idx}")
                    dense_candidates[cid] = {
                        "rank": rank + 1,
                        "score": float(sims[idx]),
                        "metadata": meta
                    }

        # Step 2: Sparse BM25 Retrieval
        sparse_scores: Dict[str, float] = {}
        sparse_ranks: Dict[str, int] = {}
        if self._bm25_model and self._corpus_cache:
            bm_scores = self._bm25_model.get_scores(query)
            ranked_bm = sorted(
                [(self._corpus_cache[i].get("chunk_id", f"chunk_{i}"), bm_scores[i], self._corpus_cache[i])
                 for i in range(len(self._corpus_cache))],
                key=lambda x: x[1],
                reverse=True
            )
            max_bm = ranked_bm[0][1] if ranked_bm and ranked_bm[0][1] > 0 else 1.0
            for rank, (cid, raw_score, meta) in enumerate(ranked_bm[:candidate_pool_size]):
                if raw_score > 0:
                    norm_score = raw_score / max_bm
                    sparse_scores[cid] = norm_score
                    sparse_ranks[cid] = rank + 1
                    if cid not in dense_candidates and (not document_filter or meta.get("document_id") == document_filter):
                        dense_candidates[cid] = {
                            "rank": candidate_pool_size + 5,
                            "score": 0.25,
                            "metadata": meta
                        }

        # Step 3: Reciprocal Rank Fusion (RRF) & Relevance blending
        rrf_k = self.settings.RRF_K
        fused_candidates: List[RetrievedDocument] = []

        for cid, dense_data in dense_candidates.items():
            dense_rank = dense_data["rank"]
            dense_score = dense_data["score"]
            meta = dense_data["metadata"]

            sparse_rank = sparse_ranks.get(cid, candidate_pool_size + 10)
            sparse_score = sparse_scores.get(cid, 0.0)

            # RRF Formula
            rrf = (1.0 / (rrf_k + dense_rank)) + (0.8 / (rrf_k + sparse_rank))
            # Blended score
            blended_score = (0.70 * dense_score) + (0.30 * sparse_score)

            doc = RetrievedDocument(
                chunk_id=cid,
                document_id=meta.get("document_id", "unknown"),
                filename=meta.get("filename", "Syllabus.pdf"),
                page=int(meta.get("page", 1)),
                unit=meta.get("unit", "General"),
                section=meta.get("section", "Overview"),
                text=meta.get("text", ""),
                score=round(blended_score, 4),
                dense_score=round(dense_score, 4),
                sparse_score=round(sparse_score, 4),
                rrf_score=round(rrf, 5)
            )
            fused_candidates.append(doc)

        fused_candidates.sort(key=lambda d: (d.score, d.rrf_score), reverse=True)
        top_candidates = fused_candidates[:top_k]

        overall_confidence = (
            sum(d.score for d in top_candidates) / len(top_candidates)
            if top_candidates else 0.0
        )

        return top_candidates, round(overall_confidence, 4)
