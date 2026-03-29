#!/usr/bin/env python3
"""
Phase 2: Embed each chunk from data/processed.json and build a local FAISS index.

Supports chunk schema:
  { "title", "content", "url" }  (preferred)
  or legacy { "title", "text", "source_url" }

Writes:
  - data/index.faiss
  - data/chunks_meta.json (aligned row order with the FAISS index)

Uses local sentence-transformers (see backend settings); no embedding API key.
Rebuild the index after changing the embedding model (dimension must match the FAISS index).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# Load environment before importing app settings
load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env")

from app.config import get_settings  # noqa: E402
from app.services.embeddings import embed_texts  # noqa: E402


def chunk_embedding_text(c: dict) -> str:
    """Plain text sent to the embedding model."""
    return (c.get("content") or c.get("text") or "").strip()


def chunk_source_url(c: dict) -> str:
    return (c.get("url") or c.get("source_url") or "").strip()


def chunk_title(c: dict) -> str:
    return (c.get("title") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from processed.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed.json",
        help="Path to Phase-1 processed chunks JSON",
    )
    parser.add_argument(
        "--out-index",
        type=Path,
        default=ROOT / "data" / "index.faiss",
        help="Output FAISS index path",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=ROOT / "data" / "chunks_meta.json",
        help="Output chunk metadata JSON (same row order as index)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Missing {args.input}. Run scripts/scrape_help.py first.")

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("chunks") or []
    if not chunks:
        raise SystemExit("No chunks found in processed.json")

    texts = []
    for c in chunks:
        t = chunk_embedding_text(c)
        if not t:
            raise SystemExit(f"Chunk missing content/text: {c.get('id', '?')}")
        texts.append(t)

    settings = get_settings()

    # Step 1: batch embeddings (L2-normalized in embed_texts)
    vectors = embed_texts(texts, settings)
    dim = vectors.shape[1]

    # Step 2: FAISS inner product on unit vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_index))

    meta = [
        {
            "id": c.get("id", str(i)),
            "text": chunk_embedding_text(c),
            "title": chunk_title(c),
            "source_url": chunk_source_url(c),
        }
        for i, c in enumerate(chunks)
    ]
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_index} ({index.ntotal} vectors, dim={dim})")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
