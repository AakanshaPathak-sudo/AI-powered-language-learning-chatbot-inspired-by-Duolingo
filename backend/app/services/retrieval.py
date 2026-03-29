"""FAISS-based similarity search over precomputed chunk embeddings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One search hit with metadata."""

    chunk_id: str
    text: str
    title: str
    source_url: str
    score: float


class FaissRetriever:
    """
    Loads a FAISS index (inner product on L2-normalized vectors = cosine similarity)
    and chunk metadata produced by scripts/build_index.py.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index: faiss.Index | None = None
        self._chunks: list[dict] = []
        self._dim: int = 0

    def load(self) -> None:
        """Read index.faiss and chunks_meta.json from data directory."""
        data_dir = Path(self._settings.data_dir)
        index_path = data_dir / self._settings.faiss_index_path
        meta_path = data_dir / self._settings.chunks_meta_path

        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing vector index. Expected {index_path} and {meta_path}. "
                "Run: python scripts/build_index.py (after scrape_help.py)."
            )

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._dim = self._index.d
        logger.info("Loaded FAISS index with %s vectors, dim=%s", self._index.ntotal, self._dim)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """
        query_vector: shape (1, dim) or (dim,); L2-normalized.
        Returns top_k chunks by cosine similarity (higher is better).
        """
        if self._index is None:
            raise RuntimeError("Retriever not loaded.")

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(f"Query dim {q.shape[1]} != index dim {self._dim}")

        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            row = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=row.get("id", str(idx)),
                    text=row["text"],
                    title=row.get("title", ""),
                    source_url=row.get("source_url", ""),
                    score=float(score),
                )
            )
        return out
