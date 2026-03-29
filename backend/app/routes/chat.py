"""Chat endpoint: embed query -> retrieve -> Groq."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, SourceRef
from app.services.chat_log import append_chat_record, build_record
from app.services.embeddings import embed_query
from app.services.rag import generate_answer
from app.services.retrieval import FaissRetriever

logger = logging.getLogger(__name__)


def _detail_to_str(detail: Any) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    try:
        return str(detail)
    except Exception:
        return repr(detail)

router = APIRouter(prefix="/chat", tags=["chat"])

# Module-level retriever; initialized in main lifespan
retriever: Optional[FaissRetriever] = None


def set_retriever(r: Optional[FaissRetriever]) -> None:
    """Inject shared retriever from app factory."""
    global retriever
    retriever = r


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    """
    RAG pipeline: embed the user query, fetch top-k similar chunks, then call Groq.
    """
    query = body.query.strip()
    record_id = str(uuid.uuid4())

    if retriever is None or not retriever.is_ready:
        append_chat_record(
            settings,
            build_record(
                query=query or body.query,
                error="Search index not loaded. Run scripts/build_index.py after scraping.",
                record_id=record_id,
            ),
        )
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Run scripts/build_index.py after scraping.",
        )

    if not query:
        append_chat_record(
            settings,
            build_record(query=body.query, error="Query must not be empty.", record_id=record_id),
        )
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        # Step 1: query embedding (normalized)
        q_vec = embed_query(query, settings)
        q_np = np.asarray(q_vec, dtype=np.float32)

        # Step 2: top-k retrieval
        hits = retriever.search(q_np, top_k=settings.top_k_retrieval)

        # Step 3: LLM with context
        answer = generate_answer(query, hits, settings)

        # Dedupe sources by URL for the UI
        seen: set[str] = set()
        sources: list[SourceRef] = []
        for h in hits:
            key = h.source_url or h.title
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceRef(title=h.title, source_url=h.source_url))

        response = ChatResponse(answer=answer, sources=sources)
        append_chat_record(
            settings,
            build_record(
                query=query,
                answer=response.answer,
                sources=[s.model_dump() for s in response.sources],
                record_id=record_id,
            ),
        )
        return response
    except ValueError as e:
        logger.warning("Configuration error: %s", e)
        append_chat_record(
            settings,
            build_record(query=query, error=_detail_to_str(e), record_id=record_id),
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat pipeline failed")
        append_chat_record(
            settings,
            build_record(
                query=query,
                error=f"Upstream model or embedding error: {e!s}",
                record_id=record_id,
            ),
        )
        raise HTTPException(status_code=502, detail="Upstream model or embedding error.") from e
