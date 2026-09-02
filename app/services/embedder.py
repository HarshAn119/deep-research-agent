"""
app/services/embedder.py

WHY tiktoken for chunking (not character count):
  - Embedding models have token limits, not character limits.
  - tiktoken gives exact token counts for OpenAI-compatible tokenizers.

EMBEDDING PROVIDERS:
  - openai/anthropic: uses OpenAI text-embedding-3-small (1536 dims, API key required)
  - ollama: uses nomic-embed-text served locally (768 dims, no API key)
    nomic-embed-text is a strong open-source embedder competitive with
    OpenAI's small model. Pull with: ollama pull nomic-embed-text

DEDUPLICATION QUERY DESIGN:
  - Query pgvector for the most similar existing chunk per new chunk.
  - If max similarity >= threshold, discard as duplicate.
  - Scoped to job_id so chunks from different jobs never interfere.
  - <=> is cosine distance (0=identical); similarity = 1 - distance.
"""

import uuid

import tiktoken
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.graph.state import ScrapedChunk
from app.models.orm import ResearchSourceChunk

_tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text_content: str) -> list[str]:
    tokens = _tokenizer.encode(text_content)
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(_tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += size - overlap
    return chunks


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds texts using the configured provider.
    - openai/anthropic → OpenAI text-embedding-3-small (1536 dims)
    - ollama → nomic-embed-text served locally (768 dims)
    """
    if settings.active_llm_provider in ("openai", "anthropic"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    # Ollama embeddings — synchronous HTTP, run in thread to avoid blocking
    import asyncio
    import httpx

    def _embed_batch() -> list[list[float]]:
        results = []
        with httpx.Client(base_url=settings.ollama_base_url, timeout=60) as client:
            for t in texts:
                resp = client.post("/api/embeddings", json={"model": settings.ollama_embedding_model, "prompt": t})
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
        return results

    return await asyncio.get_event_loop().run_in_executor(None, _embed_batch)


async def store_unique_chunks(
    job_id: uuid.UUID,
    url: str,
    title: str | None,
    raw_text: str,
    db: AsyncSession,
) -> list[ScrapedChunk]:
    """
    Chunks raw_text, embeds each chunk, deduplicates against existing job chunks
    in pgvector, and persists only unique chunks to the DB.

    Returns the list of ScrapedChunk dicts that were actually stored.
    """
    chunks = chunk_text(raw_text)
    if not chunks:
        return []

    embeddings = await embed_texts(chunks)
    stored: list[ScrapedChunk] = []

    for chunk_text_content, embedding in zip(chunks, embeddings):
        # Cosine distance query: <=> returns distance (0=identical, 2=opposite).
        # similarity = 1 - distance. We discard if similarity >= threshold.
        # ASSUMPTION: querying one-by-one per chunk is acceptable at v1 scale.
        # At high chunk volumes this could be batched, but adds complexity.
        result = await db.execute(
            text("""
                SELECT 1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM research_source_chunks
                WHERE job_id = :job_id
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 1
            """),
            {"embedding": str(embedding), "job_id": str(job_id)},
        )
        row = result.fetchone()

        if row and row.similarity >= settings.dedup_similarity_threshold:
            # Duplicate — discard to conserve context space.
            continue

        chunk_id = uuid.uuid4()
        db.add(ResearchSourceChunk(
            chunk_id=chunk_id,
            job_id=job_id,
            url=url,
            title=title,
            content_chunk=chunk_text_content,
            embedding=embedding,
        ))
        stored.append(ScrapedChunk(
            url=url,
            title=title,
            content=chunk_text_content,
            chunk_id=chunk_id,
        ))

    await db.commit()
    return stored
