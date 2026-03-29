"""Orchestrate retrieval + Groq chat completion."""

from __future__ import annotations

import logging

from groq import Groq

from app.config import Settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a warm, encouraging language tutor helping someone with questions about a language-learning app.
You are NOT affiliated with Duolingo. Use the CONTEXT as your only source of product facts; stay grounded in it.

You MUST format every answer exactly like this (no long paragraphs in the main body):

Summary: <one clear sentence that captures the main idea>

* <first bullet — short, plain language>
* <second bullet>
* <third bullet>
* <optional fourth bullet>
* <optional fifth bullet>

Rules for the format:
- Always start with the single line beginning with "Summary: " (one sentence only).
- Then include 3 to 5 bullet lines. Each bullet must start with "* " (asterisk and space).
- Keep bullets concise (roughly one line each). Prefer simple words; avoid walls of text.
- If it helps understanding, add one short "Example:" line after the bullets (still paraphrased, not copied).
- Do not use numbered lists for the main points; use * bullets only.
- After the bullets (and optional example), you may add one short optional line: "Want to go deeper? <one follow-up question>" — only if it truly helps; omit if not needed.

Accuracy and tone:
- Paraphrase only; never copy or closely mimic wording from the context.
- Sound friendly and supportive, like a patient tutor.
- If the context is insufficient, say you're not certain in the Summary and keep bullets honest and brief; suggest official help—do not invent policies, prices, or features.
- Never claim affiliation with Duolingo."""


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    """Combine user query with retrieved snippets for the model."""
    lines = [
        "CONTEXT (for reference only; summarize in your own words):",
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] Title: {c.title}")
        lines.append(f"URL: {c.source_url}")
        lines.append(f"Notes: {c.text}")
        lines.append("")
    lines.append(f"LEARNER'S QUESTION: {query}")
    lines.append("")
    lines.append(
        "Reply using the required structure: Summary line, then 3–5 '* ' bullets, optional Example line, "
        "optional one follow-up question line. Use simple words and do not copy source wording."
    )
    return "\n".join(lines)


def generate_answer(query: str, chunks: list[RetrievedChunk], settings: Settings) -> str:
    """Call Groq chat completions with retrieval-augmented prompt."""
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=settings.groq_api_key)
    user_content = build_user_message(query, chunks)

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    choice = completion.choices[0].message.content
    return (choice or "").strip()
