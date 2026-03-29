"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat payload."""

    query: str = Field(..., min_length=1, description="User question about Duolingo help topics.")


class SourceRef(BaseModel):
    """One retrieved chunk shown to the client as a citation."""

    title: str
    source_url: str


class ChatResponse(BaseModel):
    """Assistant reply with optional source references."""

    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
