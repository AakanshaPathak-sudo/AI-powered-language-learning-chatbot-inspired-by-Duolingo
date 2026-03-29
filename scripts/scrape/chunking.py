"""Token-aware chunking with sentence/paragraph boundaries (no mid-sentence cuts when avoidable)."""

from __future__ import annotations

import re

import tiktoken


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and blank lines; trim."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    min_tokens: int = 300,
    max_tokens: int = 500,
) -> list[str]:
    """
    Split into segments roughly in [min_tokens, max_tokens].
    Prefers paragraph boundaries, then sentence boundaries, then token windows as last resort.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    if count_tokens(text, enc) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_tokens = 0

    for para in paragraphs:
        pt = count_tokens(para, enc)
        if pt > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                s = sent.strip()
                if not s:
                    continue
                st = count_tokens(s, enc)
                if buf_tokens + st > max_tokens and buf_tokens >= min_tokens:
                    flush()
                if st > max_tokens:
                    ids = enc.encode(s)
                    for i in range(0, len(ids), max_tokens):
                        part = enc.decode(ids[i : i + max_tokens])
                        if part.strip():
                            chunks.append(part.strip())
                    continue
                buf.append(s)
                buf_tokens += st
                if buf_tokens >= max_tokens:
                    flush()
            continue

        if buf_tokens + pt > max_tokens and buf_tokens >= min_tokens:
            flush()
        buf.append(para)
        buf_tokens += pt
        if buf_tokens >= max_tokens:
            flush()

    flush()

    merged: list[str] = []
    for c in chunks:
        c = normalize_whitespace(c)
        if not c:
            continue
        if merged and count_tokens(c, enc) < max(min_tokens // 3, 1):
            merged[-1] = normalize_whitespace(merged[-1] + "\n\n" + c)
        else:
            merged.append(c)
    return merged if merged else [normalize_whitespace(text[:12000])]
