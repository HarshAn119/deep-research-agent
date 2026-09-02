"""
app/services/search.py

WHY Tavily as primary:
  - Tavily is purpose-built for LLM research pipelines. It returns pre-extracted
    clean text content alongside URLs, which means we often don't need to scrape
    the page at all — reducing latency and scraper blocking risk.
  - It also filters low-quality/spam results better than raw search engines.

WHY DuckDuckGo as fallback:
  - Requires no API key — zero-config fallback that works immediately.
  - Rate-limited but sufficient for fallback scenarios where Tavily fails.
  - DDGS (duckduckgo-search) returns URLs we then scrape ourselves.

RETURN CONTRACT:
  - Both providers return List[SearchResult] with the same shape.
  - Callers (search_scrape node) don't need to know which provider was used.
  - `extracted_content` is populated by Tavily (pre-extracted). When using DDG,
    it is None and the scraper service handles content extraction.
"""

from dataclasses import dataclass

from duckduckgo_search import DDGS
from tavily import TavilyClient

from app.core.config import settings


@dataclass
class SearchResult:
    url: str
    title: str | None
    extracted_content: str | None  # Pre-extracted by Tavily; None for DDG results


async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Executes a web search query. Tries Tavily first, falls back to DuckDuckGo.
    Returns up to max_results results.
    """
    if settings.tavily_api_key:
        try:
            return await _tavily_search(query, max_results)
        except Exception:
            # Tavily failed (rate limit, network error, etc.) — fall through to DDG.
            pass
    return _ddg_search(query, max_results)


async def _tavily_search(query: str, max_results: int) -> list[SearchResult]:
    """
    Tavily search with include_raw_content=True to get pre-extracted page text.
    This avoids a separate scrape step for most results.

    WHY include_raw_content=True:
      - Tavily can return the cleaned page body alongside the URL.
      - When available, this is higher quality than our own scraper output
        because Tavily's extraction is tuned for article content.
    """
    client = TavilyClient(api_key=settings.tavily_api_key)
    # TavilyClient is sync — run in executor to avoid blocking the event loop.
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
        ),
    )
    return [
        SearchResult(
            url=r["url"],
            title=r.get("title"),
            extracted_content=r.get("raw_content") or r.get("content"),
        )
        for r in response.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    """
    DuckDuckGo search — returns URLs only (no pre-extracted content).
    The scraper service will fetch content for these URLs.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(SearchResult(
                url=r["href"],
                title=r.get("title"),
                extracted_content=None,  # Must be scraped separately
            ))
    return results
