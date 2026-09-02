"""
app/graph/nodes/search_scrape.py

WHAT this node does:
  - Iterates over sub_queries from state.
  - For each query: calls search service → gets SearchResult list.
  - For each result: if Tavily provided extracted_content, use it directly.
    Otherwise scrape the URL via the scraper service.
  - Passes raw text to embedder.store_unique_chunks for chunking + dedup + storage.
  - Accumulates newly stored chunks into state.

WHY we process queries sequentially (not fully parallel):
  - Fully parallel scraping across all queries simultaneously risks hitting
    rate limits on both Tavily and target websites.
  - Within each query, results are scraped concurrently (asyncio.gather) —
    this is the right level of parallelism: fast enough, safe enough.

WHY we prefer Tavily's extracted_content over scraping:
  - Tavily's extraction is already cleaned and article-focused.
  - Scraping the same URL ourselves adds latency with no quality benefit.
  - We only scrape when extracted_content is None (DDG fallback results).

SSE: emits a search_executed event per query.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import AgentState, ScrapedChunk
from app.models.schemas import SSESearchExecutedEvent
from app.services import embedder, scraper, search


async def search_scrape_node(state: AgentState, db: AsyncSession) -> dict:
    new_chunks: list[ScrapedChunk] = []

    for query in state["sub_queries"]:
        results = await search.search(query)

        # Process results sequentially — asyncpg connections do not support
        # concurrent queries on the same session (InterfaceError otherwise).
        query_chunks: list[ScrapedChunk] = []
        for result in results:
            content = result.extracted_content or await scraper.scrape_url(result.url)
            if not content:
                continue
            chunks = await embedder.store_unique_chunks(
                job_id=state["job_id"],
                url=result.url,
                title=result.title,
                raw_text=content,
                db=db,
            )
            query_chunks.extend(chunks)

        await state["sse_queue"].put(SSESearchExecutedEvent(
            step="SEARCHING",
            query=query,
            urls_found=len(results),
        ).model_dump())

        new_chunks.extend(query_chunks)

    return {
        "all_scraped_chunks": state["all_scraped_chunks"] + new_chunks,
        "iteration": state["iteration"] + 1,
    }
