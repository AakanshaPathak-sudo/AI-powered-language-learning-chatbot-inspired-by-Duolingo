"""Chat endpoint: embed query -> retrieve -> Groq."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_log import append_chat_record, build_record
from app.services.chat_pipeline import run_rag_turn
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
        response = run_rag_turn(query, retriever, settings)
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
