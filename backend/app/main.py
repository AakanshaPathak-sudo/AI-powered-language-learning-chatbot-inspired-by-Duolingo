"""FastAPI entrypoint: CORS, lifespan loading of FAISS retriever, and routes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import chat as chat_routes
from app.services.retrieval import FaissRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vector index once at startup."""
    settings = get_settings()
    retriever = FaissRetriever(settings)
    data_dir = Path(settings.data_dir)
    index_path = data_dir / settings.faiss_index_path
    meta_path = data_dir / settings.chunks_meta_path
    logger.info("Vector index paths: %s | %s", index_path, meta_path)

    try:
        retriever.load()
        chat_routes.set_retriever(retriever)
        logger.info("Retriever ready.")
    except FileNotFoundError as e:
        logger.warning("%s — /chat will return 503 until index is built.", e)
        chat_routes.set_retriever(retriever)  # not ready; is_ready False
    except Exception as e:
        logger.exception("Failed to load FAISS index — /chat will return 503: %s", e)
        chat_routes.set_retriever(retriever)
    yield
    chat_routes.set_retriever(None)


app = FastAPI(
    title="Duolingo-style Help RAG API",
    description="Independent demo; not affiliated with Duolingo.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_routes.router)


@app.get("/health")
def health():
    """Liveness check for local development."""
    settings = get_settings()
    r = chat_routes.retriever
    data_dir = Path(settings.data_dir)
    index_path = data_dir / settings.faiss_index_path
    meta_path = data_dir / settings.chunks_meta_path
    return {
        "ok": True,
        "index_loaded": bool(r and r.is_ready),
        "data_dir": str(data_dir.resolve()),
        "index_path": str(index_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "index_file_exists": index_path.is_file(),
        "meta_file_exists": meta_path.is_file(),
        "hint": "If index_file_exists is false, run: python scripts/build_index.py then restart uvicorn.",
    }
