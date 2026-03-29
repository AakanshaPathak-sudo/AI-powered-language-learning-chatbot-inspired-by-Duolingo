"""Standard marketing pages: extract visible main text with Playwright (inner_text / locators)."""

from __future__ import annotations

import logging

from playwright.sync_api import Page

from scrape.chunking import normalize_whitespace

logger = logging.getLogger(__name__)

# Prefer regions that contain editorial content, not global chrome.
_MAIN_SELECTORS: tuple[str, ...] = (
    "main",
    '[role="main"]',
    "article",
    "#root main",
    "main:first-of-type",
    ".post-content",
    "#content",
)


def _strip_chrome_inner_text_from_element(page: Page, css_selector: str) -> str:
    """
    Clone the node in the page context and drop nav/footer/header before reading text.
    Avoids storing HTML — returns visible text only.
    """
    return page.evaluate(
        """(sel) => {
      const el = document.querySelector(sel);
      if (!el) return '';
      const clone = el.cloneNode(true);
      clone.querySelectorAll(
        'nav, footer, header, script, style, noscript, iframe, svg, [role="navigation"], [role="contentinfo"], [role="banner"]'
      ).forEach(n => n.remove());
      return (clone.innerText || '').trim();
    }""",
        css_selector,
    )


def extract_standard_main_text(page: Page, page_url: str, settle_ms: int = 2500) -> tuple[str, str]:
    """
    Return (document_title, main_body_text) using visible text only.
    Tries main/article/role=main first, then #root/body with chrome stripped.
    """
    try:
        page.wait_for_selector("main, article, [role='main'], #root, body", timeout=45_000)
    except Exception:
        logger.debug("No main/article/root selector within timeout; continuing for %s", page_url)

    page.wait_for_timeout(settle_ms)

    doc_title = normalize_whitespace(page.title() or "")

    for sel in _MAIN_SELECTORS:
        loc = page.locator(sel).first
        try:
            if loc.count() == 0:
                continue
            raw = loc.inner_text(timeout=20_000)
            text = normalize_whitespace(raw)
            if len(text) >= 120:
                logger.info(
                    "Standard page %s: extracted %d chars via locator %r",
                    page_url,
                    len(text),
                    sel,
                )
                return doc_title or page_url, text
        except Exception as e:
            logger.debug("Selector %r failed: %s", sel, e)
            continue

    for sel in ("#root", "body"):
        try:
            text = _strip_chrome_inner_text_from_element(page, sel)
            text = normalize_whitespace(text)
            if len(text) >= 120:
                logger.info(
                    "Standard page %s: extracted %d chars via %r (chrome stripped)",
                    page_url,
                    len(text),
                    sel,
                )
                return doc_title or page_url, text
        except Exception as e:
            logger.debug("Fallback %r failed: %s", sel, e)

    logger.warning("Standard page %s: insufficient text after extraction", page_url)
    return doc_title or page_url, ""


def scrape_standard_url(page: Page, url: str, settle_ms: int = 7000) -> tuple[str, str, str]:
    """
    Navigate and return (title, body_text, url). body_text is visible main content only.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        # Extra time for client-side render (SPAs)
        page.wait_for_timeout(settle_ms)
        title, text = extract_standard_main_text(page, url, settle_ms=2000)
        if len(text) < 120:
            page.wait_for_timeout(5000)
            title, text = extract_standard_main_text(page, url, settle_ms=1500)
        return title, text, url
    except Exception as e:
        logger.warning("Standard scrape failed for %s: %s", url, e)
        return "", "", url
