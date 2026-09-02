"""
app/services/scraper.py

WHY Playwright as primary scraper:
  - Many modern pages render content via JavaScript. A plain HTTP GET returns
    an empty shell. Playwright runs a real browser (headless Chromium) and
    waits for the DOM to settle before extracting text.
  - Downside: higher memory and latency per page. We only use it when
    BeautifulSoup returns insufficient content (< MIN_CONTENT_LENGTH chars).

WHY BeautifulSoup as fallback / first attempt:
  - For static HTML pages, BS4 is significantly faster and lighter than
    launching a browser context.
  - Strategy: try BS4 first via httpx. If content is too short (likely a
    JS-rendered page), escalate to Playwright.

CONTENT CLEANING:
  - We strip <script>, <style>, <nav>, <header>, <footer>, <aside> tags.
  - These contain boilerplate (ads, menus, cookie banners) that pollutes
    embeddings and wastes context window tokens.

ASSUMPTION: Playwright browser is launched once per process (singleton) and
  reused across scrape calls. Launching a new browser per request would be
  prohibitively slow (~1-2s startup per call).
"""

import asyncio
import re

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import Browser, async_playwright

# Minimum characters of extracted text to consider a scrape successful.
# Below this threshold we assume the page is JS-rendered and escalate to Playwright.
_MIN_CONTENT_LENGTH = 200

_browser: Browser | None = None
_browser_lock = asyncio.Lock()


async def _get_browser() -> Browser:
    """Lazy singleton browser — created on first scrape, reused thereafter."""
    global _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            playwright = await async_playwright().start()
            _browser = await playwright.chromium.launch(headless=True)
    return _browser


async def close_browser() -> None:
    """Called on app shutdown to cleanly close the browser process."""
    global _browser
    if _browser and _browser.is_connected():
        await _browser.close()
        _browser = None


def _clean_html(html: str) -> str:
    """
    Parses HTML and returns cleaned plain text.
    Removes script, style, nav, header, footer, aside tags before extraction.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def scrape_url(url: str) -> str | None:
    """
    Scrapes a URL and returns cleaned plain text content.
    Returns None if the page cannot be scraped (403, timeout, etc.).

    Strategy:
      1. Try httpx (fast, lightweight) + BeautifulSoup.
      2. If content too short → escalate to Playwright (handles JS rendering).
      3. If Playwright also fails → return None (caller handles gracefully).
    """
    # Step 1: Fast path via httpx + BS4
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = _clean_html(response.text)
            if len(content) >= _MIN_CONTENT_LENGTH:
                return content
    except Exception:
        pass  # Fall through to Playwright

    # Step 2: Playwright for JS-rendered pages
    try:
        browser = await _get_browser()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            html = await page.content()
            content = _clean_html(html)
            return content if len(content) >= _MIN_CONTENT_LENGTH else None
        finally:
            await page.close()
    except Exception:
        return None
