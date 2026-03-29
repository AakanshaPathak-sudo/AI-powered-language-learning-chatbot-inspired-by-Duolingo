#!/usr/bin/env python3
"""
Scrape curated public Duolingo URLs into chunked data/processed.json.

Uses Playwright for rendering and visible-text extraction (`inner_text` / locators) on
standard pages, plus dedicated logic for the expandable /help FAQ hub.

Independent educational project — not affiliated with Duolingo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ROOT = Path(__file__).resolve().parents[1]

from playwright.sync_api import sync_playwright

from scrape.chunking import chunk_text, count_tokens, get_encoder, normalize_whitespace
from scrape.help_faq import scrape_help_faq_entries
from scrape.standard_page import scrape_standard_url
from scrape.urls_loader import PageKind, load_url_entries

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("scrape_help")


def fingerprint_chunk(url: str, content: str) -> str:
    """Stable key for deduplication."""
    return hashlib.sha256(
        f"{url}\n{normalize_whitespace(content)}".encode("utf-8")
    ).hexdigest()


def documents_to_chunks(
    documents: list[dict[str, str]],
    enc,
    seen_fps: set[str],
) -> list[dict]:
    """Turn {title, content, url} rows into 300–500 token chunks; dedupe."""
    out: list[dict] = []
    chunk_idx = 0
    for doc in documents:
        title = doc.get("title") or ""
        url = doc.get("url") or ""
        raw = doc.get("content") or ""
        raw = normalize_whitespace(raw)
        if len(raw) < 40:
            logger.warning("Skip document with very short content (%d chars): %s", len(raw), title[:80])
            continue
        logger.info("Document %r: source text length %d chars before chunking", title[:72], len(raw))
        parts = chunk_text(raw, enc, 300, 500)
        for part in parts:
            fp = fingerprint_chunk(url, part)
            if fp in seen_fps:
                logger.info("Skip duplicate chunk (%s…)", part[:48])
                continue
            seen_fps.add(fp)
            out.append(
                {
                    "id": f"chunk_{chunk_idx}",
                    "title": title,
                    "content": part,
                    "url": url,
                    "token_count": count_tokens(part, enc),
                }
            )
            logger.info(
                "Chunk %d: %d chars (~%d tokens) url=%s",
                chunk_idx,
                len(part),
                count_tokens(part, enc),
                url[:60],
            )
            chunk_idx += 1
    return out


def run_scrape(urls_path: Path | None, out_path: Path) -> list[dict]:
    enc = get_encoder()
    entries = load_url_entries(urls_path, ROOT)
    seen_fps: set[str] = set()
    all_chunks: list[dict] = []

    all_docs: list[dict[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for entry in entries:
            url = entry["url"]
            kind: PageKind = entry.get("kind") or "standard"
            logger.info("Scraping %s (%s)", url, kind)

            try:
                if kind == "help_faq":
                    all_docs.extend(scrape_help_faq_entries(page))
                else:
                    title, text, u = scrape_standard_url(page, url)
                    if text:
                        combined = normalize_whitespace(f"{title}\n\n{text}" if title else text)
                        all_docs.append({"title": title or u, "content": combined, "url": u})
                    else:
                        logger.warning("No body text for %s", url)
            except Exception as e:
                logger.warning("Failed %s — skipping: %s", url, e)
                continue

            logger.info("Collected %d document(s) so far (last URL %s)", len(all_docs), url)

        browser.close()

    logger.info("Merged %d source documents before chunking", len(all_docs))
    all_chunks = documents_to_chunks(all_docs, enc, seen_fps)

    payload = {
        "meta": {
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Independent educational project; not affiliated with Duolingo.",
            "token_encoding": "cl100k_base",
            "chunk_token_range": [300, 500],
        },
        "chunks": all_chunks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %s — %d total chunks", out_path, len(all_chunks))
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape curated URLs into data/processed.json")
    parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        help="Path to urls.json (default: data/urls.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dry-sample",
        action="store_true",
        help="Write a minimal processed.json without network access",
    )
    args = parser.parse_args()

    if args.dry_sample:
        sample = {
            "meta": {
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "disclaimer": "No network run (--dry-sample). Chunks are empty; run `python scripts/scrape_help.py` for real content.",
            },
            "chunks": [],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        logger.info("Wrote empty dry sample to %s (no placeholder text)", args.out)
        return

    run_scrape(args.urls, args.out)


if __name__ == "__main__":
    main()
