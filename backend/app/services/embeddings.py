"""Local text embeddings via sentence-transformers (no external embedding API)."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Lazy singletons keyed by model name (thread-safe).
_models: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()


def _load_model(model_name: str) -> SentenceTransformer:
    try:
        logger.info(
            "Loading SentenceTransformer model %r (first run may download weights)",
            model_name,
        )
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.exception(
            "Failed to load sentence-transformers model %r — check disk space and network on first download.",
            model_name,
        )
        raise RuntimeError(
            f"Could not load embedding model {model_name!r}: {e}"
        ) from e


def _get_model(model_name: str) -> SentenceTransformer:
    with _model_lock:
        if model_name not in _models:
            _models[model_name] = _load_model(model_name)
        return _models[model_name]


def embed_texts(texts: list[str], settings: Optional[Settings] = None) -> np.ndarray:
    """
    Embed strings with a local model; L2-normalize rows for FAISS inner-product (cosine) search.

    Returns float32 array of shape (len(texts), dim). Empty input returns shape (0, 384) for
    all-MiniLM-L6-v2 (dim is fixed for that model).
    """
    if settings is None:
        settings = get_settings()

    model_name = settings.sentence_transformer_model

    if not texts:
        dim = _get_model(model_name).get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    model = _get_model(model_name)
    show_bar = len(texts) > 1

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=show_bar,
        normalize_embeddings=False,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(query: str, settings: Settings) -> np.ndarray:
    """Single query embedding as row vector (1, dim)."""
    return embed_texts([query], settings)
