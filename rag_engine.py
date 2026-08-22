"""
rag_engine.py - Core RAG Engine for Talk-to-Syllabus (TTS-Cloud).
Handles conversational query contextualization, grounded prompt generation with citations,
RAG guardrails, Groq LLM execution, and academic study features.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq

from config import RAGSettings, get_api_keys
from retriever import RetrievedDocument, SyllabusRetriever


@dataclass
class RAGResponse:
    """Complete structured response from the RAG engine."""
    answer: str
    standalone_query: str
    retrieved_documents: List[RetrievedDocument]
    citations: List[Dict[str, Any]]
    confidence_score: float
    is_low_confidence: bool
    latency_ms: float


class SyllabusRAGEngine:
    """
    Production RAG engine for academic syllabus question answering and study assistance.
    """

    def __init__(
        self,
        retriever: Optional[SyllabusRetriever] = None,
        groq_api_key: Optional[str] = None
    ):
        self.settings = RAGSettings()
        _, key, _ = get_api_keys()
        self.groq_api_key = groq_api_key or key
        self.retriever = retriever or SyllabusRetriever()
        self._groq_client = None

    def _get_groq_client(self) -> Groq:
        """Initialize or retrieve cached Groq client."""
        if self._groq_client is None:
            if not self.groq_api_key:
                raise ValueError(
                    "❌ Groq API Key is missing. Please set GROQ_API_KEY in .env "
                    "or pass it in the application sidebar."
                )
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    def reformulate_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Contextualize conversational follow-ups into a standalone retrieval query.
        Example:
            History: [User: "What is normalization?", Assistant: "..."]
            Query: "Explain the third one" -> Reformulated: "Explain Third Normal Form (3NF) in database normalization"
        """
        if not chat_history:
            return query

        # Extract only recent conversation turns
        recent_history = chat_history[-(self.settings.MAX_HISTORY_TURNS * 2):]
        if not recent_history:
            return query

        history_str = ""
        for msg in recent_history:
            role = "Student" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").replace("\n", " ")[:250]
            history_str += f"{role}: {content}\n"

        system_prompt = (
            "You are an academic query reformulator. Given a chat history between a student and an assistant, "
            "and a follow-up question, rewrite the question into a standalone, keyword-rich search query "
            "suitable for retrieving information from a syllabus or academic textbook. "
            "Preserve specific technical terms, unit numbers, and subject names. "
            "DO NOT answer the question. Return ONLY the reformulated query text."
        )

        user_prompt = (
            f"Conversation History:\n{history_str}\n"
            f"Latest Student Question: {query}\n\n"
            f"Standalone Search Query:"
        )

        try:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=self.settings.GROQ_REFORMULATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.0
            )
            reformulated = response.choices[0].message.content.strip()
            # If model returned something sensible and not too long
            if reformulated and len(reformulated) > 3 and len(reformulated) < 300:
                # Remove quotes if wrapped
                return reformulated.strip('"\'')
            return query
        except Exception:
            # Fallback gracefully to original query on any network or API issue
            return query

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        document_filter: Optional[str] = None,
        top_k: int = 4
    ) -> RAGResponse:
        """
        Full RAG pipeline:
        1. Contextualize query with conversation memory.
        2. Hybrid retrieve top candidates from Pinecone + BM25 with RRF.
        3. Apply confidence guardrails.
        4. Generate grounded answer with precise page and unit citations.
        """
        start_time = time.time()
        chat_history = chat_history or []

        # Step 1: Query Reformulation
        standalone_query = self.reformulate_query(question, chat_history)

        # Step 2: Hybrid Retrieval & Reranking
        docs, confidence = self.retriever.hybrid_search(
            query=standalone_query,
            document_filter=document_filter,
            top_k=top_k,
            candidate_pool_size=self.settings.INITIAL_TOP_K
        )

        is_low_confidence = confidence < self.settings.CONFIDENCE_THRESHOLD or len(docs) == 0

        # Step 3: Format Context and Structured Citations
        citations = []
        context_blocks = []
        for i, doc in enumerate(docs, start=1):
            header = f"[Source {i}: {doc.filename} | Page {doc.page} | {doc.unit} | {doc.section}]"
            context_blocks.append(f"{header}\n{doc.text}\n")
            citations.append({
                "source_id": i,
                "filename": doc.filename,
                "page": doc.page,
                "unit": doc.unit,
                "section": doc.section,
                "score": doc.score,
                "snippet": doc.text[:220] + "..." if len(doc.text) > 220 else doc.text
            })

        full_context = "\n---\n".join(context_blocks)

        # Step 4: Strict Guardrail Prompt
        system_prompt = (
            "You are TTS-Cloud (Talk-to-Syllabus), an authoritative academic course assistant.\n"
            "Your objective is to provide precise, structured, student-friendly answers based strictly on "
            "the provided syllabus context.\n\n"
            "MANDATORY RULES:\n"
            "1. GROUNDING: Base your answer EXCLUSIVELY on the provided syllabus context. Never invent topics, prerequisites, marks, or course outcomes.\n"
            "2. OUT-OF-SYLLABUS HANDLING: If the context does not contain enough information to answer the question, clearly state: "
            "'Based on the uploaded syllabus documents, this information is not available.'\n"
            "3. SOURCE CITATIONS: Whenever stating specific facts, modules, prerequisites, or topics, cite the source using: [Filename, Page X, Unit Y].\n"
            "4. CLARITY: Use bullet points, bold headings, and clean structure for ease of reading."
        )

        user_prompt = (
            f"=== RETRIEVED SYLLABUS CONTEXT ===\n"
            f"{full_context if full_context.strip() else 'No relevant syllabus context was found.'}\n"
            f"====================================\n\n"
            f"Student Question: {question}\n"
            f"(Contextualized Search: {standalone_query})\n\n"
            f"Provide a clear, grounded academic answer with explicit source citations:"
        )

        # If completely empty context or zero confidence
        if is_low_confidence and not docs:
            answer = (
                "⚠️ **No matching information found in the syllabus.**\n\n"
                "The question asked does not appear to match any topics, units, or policies in the uploaded syllabus PDF. "
                "Please verify if the syllabus for this course is uploaded or try rephrasing your question."
            )
        else:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=self.settings.GROQ_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.settings.MAX_OUTPUT_TOKENS,
                temperature=self.settings.DEFAULT_TEMPERATURE
            )
            answer = response.choices[0].message.content.strip()

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return RAGResponse(
            answer=answer,
            standalone_query=standalone_query,
            retrieved_documents=docs,
            citations=citations,
            confidence_score=confidence,
            is_low_confidence=is_low_confidence,
            latency_ms=latency_ms
        )

    # ─── Academic Syllabus Tools ─────────────────────────────────────────────

    def generate_unit_summary(self, unit_name_or_query: str, document_filter: Optional[str] = None) -> str:
        """Generate a comprehensive, structured syllabus summary for a specific unit/module."""
        search_query = f"{unit_name_or_query} topics syllabus syllabus objectives contents"
        docs, _ = self.retriever.hybrid_search(
            query=search_query,
            document_filter=document_filter,
            top_k=6,
            candidate_pool_size=10
        )
        if not docs:
            return "❌ No context found for the specified unit in the uploaded syllabus."

        context = "\n\n".join([d.text for d in docs])
        prompt = (
            f"Context from syllabus:\n{context}\n\n"
            f"Task: Generate a comprehensive, student-ready Academic Unit Summary for: '{unit_name_or_query}'.\n"
            f"Structure your response with:\n"
            f"1. 🎯 Unit Title & Objective\n"
            f"2. 📋 Core Topics Covered (in logical sequence)\n"
            f"3. 🔑 Key Concepts & Algorithms/Theorems\n"
            f"4. 💡 Learning Outcomes\n"
            f"Include exact syllabus citations where relevant."
        )

        client = self._get_groq_client()
        resp = client.chat.completions.create(
            model=self.settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()

    def generate_practice_mcqs(
        self,
        topic_or_unit: str,
        num_questions: int = 4,
        document_filter: Optional[str] = None
    ) -> str:
        """Generate high-yield academic practice MCQs with explanations and source citations."""
        docs, _ = self.retriever.hybrid_search(
            query=topic_or_unit,
            document_filter=document_filter,
            top_k=5,
            candidate_pool_size=8
        )
        if not docs:
            return "❌ No relevant syllabus content found to generate MCQs."

        context = "\n\n".join([d.text for d in docs])
        prompt = (
            f"Context from syllabus:\n{context}\n\n"
            f"Task: Create {num_questions} high-quality academic Multiple Choice Questions (MCQs) for university exams on: '{topic_or_unit}'.\n"
            f"Format strictly as:\n"
            f"Q[number]. [Question text]\n"
            f"A) [Option]\n"
            f"B) [Option]\n"
            f"C) [Option]\n"
            f"D) [Option]\n"
            f"**Correct Answer:** [Letter]\n"
            f"**Explanation & Syllabus Reference:** [Brief explanation with topic/page]\n\n"
        )

        client = self._get_groq_client()
        resp = client.chat.completions.create(
            model=self.settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a university exam paper setter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()

    def extract_key_exam_topics(self, document_filter: Optional[str] = None) -> str:
        """Analyze the syllabus to identify important exam topics, grading patterns, and core modules."""
        docs, _ = self.retriever.hybrid_search(
            query="units modules chapters topics evaluation scheme exam marks weightage",
            document_filter=document_filter,
            top_k=8,
            candidate_pool_size=12
        )
        if not docs:
            return "❌ No syllabus context available."

        context = "\n\n".join([d.text for d in docs])
        prompt = (
            f"Context from syllabus:\n{context}\n\n"
            f"Task: Analyze the uploaded syllabus and output an 'Important Exam Topics & Preparation Strategy' guide.\n"
            f"Include:\n"
            f"1. 🌟 High-Weightage Core Topics per Unit\n"
            f"2. ⚖️ Evaluation & Marking Scheme (if specified in syllabus)\n"
            f"3. ⚠️ Common Difficult Concepts to Focus On\n"
            f"4. 📖 Recommended Reference Books from the syllabus"
        )

        client = self._get_groq_client()
        resp = client.chat.completions.create(
            model=self.settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an academic mentor helping engineering and science students ace their exams."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
