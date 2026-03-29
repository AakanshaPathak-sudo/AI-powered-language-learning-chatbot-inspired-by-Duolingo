"""Only these URLs may be fetched — do not scrape beyond the curated list."""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Lowercase host/scheme; strip trailing slash from path (empty path stays empty)."""
    p = urlparse(url.strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/") or ""
    return f"{scheme}://{netloc}{path}"


ALLOWED_NORMALIZED: frozenset[str] = frozenset(
    {
        normalize_url(u)
        for u in (
            "https://www.duolingo.com/info",
            "https://www.duolingo.com/approach",
            "https://www.duolingo.com/efficacy",
            "https://blog.duolingo.com/handbook/",
            "https://design.duolingo.com",
            "https://www.duolingo.com/help",
        )
    }
)


def is_url_allowed(url: str) -> bool:
    return normalize_url(url) in ALLOWED_NORMALIZED


def assert_url_allowed(url: str) -> None:
    if not is_url_allowed(url):
        raise ValueError(f"URL not in curated allowlist: {url}")
