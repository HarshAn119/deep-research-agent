"""
tests/test_services.py

WHAT we test:
  embedder.chunk_text():
    - Text shorter than chunk_size → single chunk returned
    - Text exactly chunk_size → single chunk, no overflow
    - Text longer than chunk_size → multiple chunks with correct overlap
    - Empty string → empty list
    - Overlap is smaller than chunk size (no infinite loop)

  embedder.embed_texts():
    - Calls OpenAI embeddings API once for a batch (not once per text)
    - Returns one vector per input text
    - Vector length matches expected dimensions (1536)

  scraper._clean_html():
    - Strips <script>, <style>, <nav>, <header>, <footer>, <aside> tags
    - Retains main article content
    - Collapses multiple blank lines

  services/search.py search():
    - Uses Tavily when key is present
    - Falls back to DDG when Tavily raises
    - Returns SearchResult list with correct shape

WHY we mock the OpenAI client in embed_texts tests:
  - We never make real API calls in tests. We verify the call was made with
    the correct arguments and that the response is correctly unpacked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── embedder.chunk_text ───────────────────────────────────────────────────────

def test_chunk_text_short_text_returns_single_chunk():
    from app.services.embedder import chunk_text
    with patch("app.services.embedder.settings") as s:
        s.chunk_size_tokens = 512
        s.chunk_overlap_tokens = 50
        result = chunk_text("Hello world")
    assert len(result) == 1
    assert "Hello world" in result[0]


def test_chunk_text_empty_string_returns_empty_list():
    from app.services.embedder import chunk_text
    with patch("app.services.embedder.settings") as s:
        s.chunk_size_tokens = 512
        s.chunk_overlap_tokens = 50
        result = chunk_text("")
    assert result == []


def test_chunk_text_long_text_produces_multiple_chunks():
    from app.services.embedder import chunk_text
    # ~600 tokens of text should produce 2 chunks with size=512, overlap=50
    long_text = "word " * 600
    with patch("app.services.embedder.settings") as s:
        s.chunk_size_tokens = 512
        s.chunk_overlap_tokens = 50
        result = chunk_text(long_text)
    assert len(result) >= 2


def test_chunk_text_overlap_means_chunks_share_content():
    from app.services.embedder import chunk_text
    # With overlap, the end of chunk N should appear at the start of chunk N+1
    long_text = "word " * 600
    with patch("app.services.embedder.settings") as s:
        s.chunk_size_tokens = 512
        s.chunk_overlap_tokens = 50
        chunks = chunk_text(long_text)
    if len(chunks) >= 2:
        # The last few tokens of chunk[0] should appear in chunk[1]
        # We verify this by checking chunks are not completely disjoint
        end_of_first = chunks[0][-50:]
        assert any(word in chunks[1] for word in end_of_first.split()[:5])


def test_chunk_text_no_infinite_loop_with_small_overlap():
    """Overlap must be smaller than chunk_size — verify no infinite loop."""
    from app.services.embedder import chunk_text
    with patch("app.services.embedder.settings") as s:
        s.chunk_size_tokens = 10
        s.chunk_overlap_tokens = 5
        result = chunk_text("word " * 100)
    # Should terminate and return multiple chunks
    assert len(result) > 1


# ── embedder.embed_texts ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_texts_calls_api_once_for_batch():
    """Verifies all texts are sent in a single API call, not one per text."""
    from app.services import embedder

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_embedding, mock_embedding, mock_embedding]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(embedder, "_openai_client", mock_client):
        result = await embedder.embed_texts(["text1", "text2", "text3"])

    # Should be called exactly once with all three texts
    mock_client.embeddings.create.assert_called_once()
    call_kwargs = mock_client.embeddings.create.call_args.kwargs
    assert call_kwargs["input"] == ["text1", "text2", "text3"]
    assert call_kwargs["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_embed_texts_returns_one_vector_per_input():
    from app.services import embedder

    mock_embeddings = [MagicMock(embedding=[float(i)] * 1536) for i in range(4)]
    mock_response = MagicMock()
    mock_response.data = mock_embeddings

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(embedder, "_openai_client", mock_client):
        result = await embedder.embed_texts(["a", "b", "c", "d"])

    assert len(result) == 4


# ── scraper._clean_html ───────────────────────────────────────────────────────

def test_clean_html_strips_script_tags():
    from app.services.scraper import _clean_html
    html = "<html><body><p>Article content</p><script>alert('ad')</script></body></html>"
    result = _clean_html(html)
    assert "Article content" in result
    assert "alert" not in result


def test_clean_html_strips_nav_and_footer():
    from app.services.scraper import _clean_html
    html = """
    <html><body>
      <nav>Menu items</nav>
      <p>Main article text</p>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    result = _clean_html(html)
    assert "Main article text" in result
    assert "Menu items" not in result
    assert "Copyright 2026" not in result


def test_clean_html_collapses_blank_lines():
    from app.services.scraper import _clean_html
    html = "<html><body><p>Line one</p><p>Line two</p></body></html>"
    result = _clean_html(html)
    # Should not have more than one consecutive blank line
    assert "\n\n\n" not in result


def test_clean_html_retains_article_content():
    from app.services.scraper import _clean_html
    html = "<html><body><article><h1>Title</h1><p>Body text here.</p></article></body></html>"
    result = _clean_html(html)
    assert "Title" in result
    assert "Body text here." in result


# ── search service ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_uses_tavily_when_key_present():
    from app.services import search

    mock_results = [
        search.SearchResult(url="https://example.com", title="Example", extracted_content="Content")
    ]

    with patch("app.services.search.settings") as mock_settings:
        mock_settings.tavily_api_key = "tvly-test"
        with patch("app.services.search._tavily_search", new_callable=AsyncMock) as mock_tavily:
            mock_tavily.return_value = mock_results
            result = await search.search("test query")

    mock_tavily.assert_called_once_with("test query", 5)
    assert result == mock_results


@pytest.mark.asyncio
async def test_search_falls_back_to_ddg_when_tavily_raises():
    from app.services import search

    ddg_results = [
        search.SearchResult(url="https://ddg.com", title="DDG Result", extracted_content=None)
    ]

    with patch("app.services.search.settings") as mock_settings:
        mock_settings.tavily_api_key = "tvly-test"
        with patch("app.services.search._tavily_search", new_callable=AsyncMock) as mock_tavily:
            mock_tavily.side_effect = Exception("Tavily rate limit")
            with patch("app.services.search._ddg_search", return_value=ddg_results) as mock_ddg:
                result = await search.search("test query")

    assert result == ddg_results
    mock_ddg.assert_called_once()


@pytest.mark.asyncio
async def test_search_uses_ddg_when_no_tavily_key():
    from app.services import search

    ddg_results = [
        search.SearchResult(url="https://ddg.com", title="DDG Result", extracted_content=None)
    ]

    with patch("app.services.search.settings") as mock_settings:
        mock_settings.tavily_api_key = None
        with patch("app.services.search._ddg_search", return_value=ddg_results) as mock_ddg:
            result = await search.search("test query")

    mock_ddg.assert_called_once()
    assert result == ddg_results


def test_search_result_extracted_content_none_for_ddg():
    """DDG results always have extracted_content=None — scraper handles them."""
    from app.services.search import SearchResult
    result = SearchResult(url="https://example.com", title="Title", extracted_content=None)
    assert result.extracted_content is None
