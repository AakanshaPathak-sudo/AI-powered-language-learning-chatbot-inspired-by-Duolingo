"""Shared RAG orchestration: embed → retrieve → Groq (used by API and Streamlit)."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.schemas import ChatResponse, SourceRef
from app.services.embeddings import embed_query
from app.services.rag import generate_answer
from app.services.retrieval import FaissRetriever


def run_rag_turn(query: str, retriever: FaissRetriever, settings: Settings) -> ChatResponse:
    """
    Embed query, retrieve top-k chunks, generate answer with Groq.
    Raises ValueError for empty query, RuntimeError if index not ready.
    """
    q = query.strip()
    if not q:
        raise ValueError("Query must not be empty.")
    if not retriever.is_ready:
        raise RuntimeError(
            "Search index not loaded. Run scripts/build_index.py after scraping."
        )

    q_vec = embed_query(q, settings)
    q_np = np.asarray(q_vec, dtype=np.float32)
    hits = retriever.search(q_np, top_k=settings.top_k_retrieval)
    answer = generate_answer(q, hits, settings)

    seen: set[str] = set()
    sources: list[SourceRef] = []
    for h in hits:
        key = h.source_url or h.title
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceRef(title=h.title, source_url=h.source_url))

    return ChatResponse(answer=answer, sources=sources)
