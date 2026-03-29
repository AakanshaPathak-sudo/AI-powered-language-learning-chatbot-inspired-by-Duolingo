"""Expandable FAQ hub at www.duolingo.com/help — Playwright clicks + visible text extraction."""

from __future__ import annotations

import logging
import re
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from scrape.chunking import normalize_whitespace

logger = logging.getLogger(__name__)

HELP_URL = "https://www.duolingo.com/help"

_FALLBACK_TITLES: list[str] = [
    "Why did my course change?",
    "What is a streak?",
    "What are leaderboards and leagues?",
    "Does Duolingo use any open source libraries?",
    "How do I change my username or email address?",
    "How do I find, follow, and block users on Duolingo?",
    "How do I remove or reset a course?",
    "I’m having trouble accessing my account.",
    "How do I delete my account and access my data?",
    "What to do if your data was compromised",
    "What is Super Duolingo and how do I subscribe?",
    "Family Plan",
    "How do I cancel my Super Duolingo subscription?",
    "How do I request a refund?",
    "How do I use a promo code?",
    "What is Duolingo Max?",
]

_SKIP_TITLES = frozenset(
    {
        "Still unsure about something?",
        "SEND FEEDBACK",
        "Send Feedback",
        "READ MORE",
    }
)

_SECTION_LABELS = frozenset(
    {
        "Using Duolingo",
        "Account Management",
        "Subscription & Payments",
        "Frequently Asked Questions",
        "HELP CENTER",
        "HOME",
        "Home",
    }
)


def slugify(title: str) -> str:
    s = title.lower().replace("'", "").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "faq"


def faq_url(question: str) -> str:
    return f"{HELP_URL}#{slugify(question)}"


def title_click_variants(title: str) -> list[str]:
    variants = [title, title.replace("'", "'"), title.replace("'", "'")]
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def discover_faq_questions(page: Page) -> list[str]:
    """Discover FAQ question strings from visible layout + JS leaf nodes (questions ending in ?)."""
    body = page.inner_text("body")
    lines = [normalize_whitespace(x) for x in body.split("\n") if x.strip()]
    from_lines: list[str] = []
    for line in lines:
        if line in _SECTION_LABELS or line in _SKIP_TITLES:
            continue
        if len(line) > 120:
            continue
        if line.endswith("?"):
            from_lines.append(line)
        elif line in ("Family Plan",):
            from_lines.append(line)

    js_titles = page.evaluate(
        """() => {
      const out = [];
      const els = document.querySelectorAll('*');
      for (const el of els) {
        if (el.children.length) continue;
        const t = (el.textContent || '').trim();
        if (t && t.endsWith('?') && t.length < 120) out.push(t);
      }
      return [...new Set(out)];
    }"""
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for t in from_lines + list(js_titles):
        if t in _SKIP_TITLES or t in _SECTION_LABELS:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    if len(ordered) < 8:
        for t in _FALLBACK_TITLES:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def extract_answer_after_click(full_body: str, question: str, later_questions: list[str]) -> str:
    i = full_body.find(question)
    if i < 0:
        for alt in title_click_variants(question):
            if alt != question:
                i = full_body.find(alt)
                if i >= 0:
                    question = alt
                    break
    if i < 0:
        return ""

    rest = full_body[i + len(question) :]
    stop = len(rest)
    if "READ MORE" in rest:
        stop = min(stop, rest.index("READ MORE"))
    for nxt in later_questions:
        j = rest.find(nxt)
        if j >= 0 and j < stop:
            stop = j
    for lab in _SECTION_LABELS:
        j = rest.find(lab)
        if j >= 0 and j < stop and j > 20:
            stop = min(stop, j)
    return normalize_whitespace(rest[:stop])


def _wait_for_answer_text(page: Page, question: str, timeout_ms: int = 25_000) -> None:
    """Wait until body has substantial text after the question (answer expanded)."""
    try:
        page.wait_for_function(
            """(q) => {
          const t = document.body ? document.body.innerText : '';
          const i = t.indexOf(q);
          if (i < 0) return false;
          const tail = t.slice(i + q.length).trimStart();
          return tail.length > 80;
        }""",
            arg=question,
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        logger.warning("Timeout waiting for expanded answer after %r", question)


def scrape_help_faq_entries(
    page: Page,
    click_delay_ms: int = 400,
    settle_ms: int = 900,
    initial_wait_ms: int = 9000,
) -> list[dict[str, str]]:
    """
    For each FAQ: reload /help, click question, wait for answer, read body inner_text.
    Returns {title, content, url} with real answer text only.
    """
    page.goto(HELP_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(initial_wait_ms)
    questions = discover_faq_questions(page)
    logger.info("Help FAQ: discovered %d question rows", len(questions))

    entries: list[dict[str, str]] = []
    for idx, q in enumerate(questions):
        if q in _SKIP_TITLES:
            continue
        try:
            page.goto(HELP_URL, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(initial_wait_ms)
            page.wait_for_timeout(click_delay_ms)

            clicked = q
            last_err: Optional[Exception] = None
            for variant in title_click_variants(q):
                try:
                    loc = page.get_by_text(variant, exact=True).first
                    loc.scroll_into_view_if_needed(timeout=15_000)
                    page.wait_for_timeout(200)
                    loc.click(timeout=20_000)
                    clicked = variant
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if last_err is not None:
                logger.warning("Could not expand FAQ %r: %s", q, last_err)
                continue

            _wait_for_answer_text(page, clicked)
            page.wait_for_timeout(settle_ms)
            page.wait_for_timeout(click_delay_ms)

            body = page.inner_text("body")
            later = [x for x in questions[idx + 1 :] if x not in _SKIP_TITLES]
            answer = extract_answer_after_click(body, clicked, later)

            if not answer or len(answer) < 5:
                logger.warning("Empty answer for FAQ %r", q)
                continue

            logger.info(
                "FAQ %r: extracted answer length %d chars",
                clicked[:60],
                len(answer),
            )

            entries.append(
                {
                    "title": f"FAQ: {clicked}",
                    "content": answer,
                    "url": faq_url(clicked),
                }
            )
        except Exception as e:
            logger.warning("FAQ scrape error for %r: %s", q, e)
            continue

    return entries
