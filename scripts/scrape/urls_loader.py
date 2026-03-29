"""Load target URLs from data/urls.json or fall back to built-in defaults."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

from scrape.allowlist import assert_url_allowed, normalize_url

logger = logging.getLogger(__name__)

PageKind = Literal["standard", "help_faq"]


class UrlEntry(TypedDict, total=False):
    url: str
    kind: PageKind


DEFAULT_ENTRIES: list[UrlEntry] = [
    {"url": "https://www.duolingo.com/info", "kind": "standard"},
    {"url": "https://www.duolingo.com/approach", "kind": "standard"},
    {"url": "https://www.duolingo.com/efficacy", "kind": "standard"},
    {"url": "https://blog.duolingo.com/handbook/", "kind": "standard"},
    {"url": "https://design.duolingo.com", "kind": "standard"},
    {"url": "https://www.duolingo.com/help", "kind": "help_faq"},
]


def infer_kind(url: str) -> PageKind:
    n = normalize_url(url)
    if n.endswith("duolingo.com/help") or n.endswith("/help"):
        return "help_faq"
    return "standard"


def load_url_entries(path: Path | None, repo_root: Path) -> list[UrlEntry]:
    """Load and validate URLs; every URL must pass the allowlist."""
    if path is None:
        path = repo_root / "data" / "urls.json"

    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("urls") or data.get("entries") or []
        entries: list[UrlEntry] = []
        for item in raw:
            if isinstance(item, str):
                u = item
                k = infer_kind(u)
            else:
                u = item.get("url") or item.get("href")
                k = item.get("kind") or infer_kind(u)
            if not u:
                continue
            assert_url_allowed(u)
            entries.append({"url": u.strip(), "kind": k if k in ("standard", "help_faq") else infer_kind(u)})
        logger.info("Loaded %d URLs from %s", len(entries), path)
        return entries

    logger.info("No %s — using built-in default URL list", path)
    for e in DEFAULT_ENTRIES:
        assert_url_allowed(e["url"])
    return list(DEFAULT_ENTRIES)
