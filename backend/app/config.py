"""Central settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    # Load `.env` from the backend folder or the repository root when running locally.
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Local embeddings (sentence-transformers); no API key required
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    top_k_retrieval: int = 5

    # Append-only JSONL log of queries and responses (under data_dir)
    chat_log_enabled: bool = True
    chat_log_path: str = "chat_log.jsonl"

    # Paths relative to repo root when running from backend/; overridden by env if needed
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    faiss_index_path: str = "index.faiss"
    chunks_meta_path: str = "chunks_meta.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
