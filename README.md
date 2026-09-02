# Autonomous Deep Research Agent

A backend service that autonomously researches any topic by searching the live web, scraping content, deduplicating it with vector similarity, reflecting on knowledge gaps, and synthesising a fully cited Markdown report — all while streaming live progress via Server-Sent Events.

For full architecture, flow, and design decisions see **[docs/PROJECT.md](docs/PROJECT.md)**.

---

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with the [pgvector extension](https://github.com/pgvector/pgvector)
- An OpenAI API key (or Anthropic)
- A Tavily API key (optional but recommended)

---

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <repo-url>
cd deep-research-agent
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -e ".[dev]"
```

**3. Install Playwright browsers**
```bash
playwright install chromium
```

**4. Configure environment**
```bash
cp .env.example .env
# Edit .env and fill in your API keys and DATABASE_URL
```

**5. Set up the database**

Create the database and enable pgvector:
```sql
CREATE DATABASE deep_research;
\c deep_research
CREATE EXTENSION IF NOT EXISTS vector;
```

Run migrations:
```bash
alembic upgrade head
```

---

## Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Usage

**Submit a research job:**
```bash
curl -X POST http://localhost:8000/api/v1/research/jobs \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Impact of Rust on backend system performance vs Go in 2026",
    "depth": "deep"
  }'
```

Response:
```json
{
  "job_id": "8f3b2a9d-...",
  "status": "QUEUED",
  "created_at": "2026-01-01T00:00:00Z",
  "sse_stream_url": "/api/v1/research/jobs/8f3b2a9d-.../stream"
}
```

**Stream live progress:**
```bash
curl -N http://localhost:8000/api/v1/research/jobs/8f3b2a9d-.../stream \
  -H "X-API-Key: your-api-key"
```

**Fetch the final report:**
```bash
curl http://localhost:8000/api/v1/research/jobs/8f3b2a9d-... \
  -H "X-API-Key: your-api-key"
```

---

## Request Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `topic` | string | Yes | Research topic (10–2000 chars) |
| `depth` | `fast` \| `deep` | No | `fast` = 2 iterations, 3 queries. `deep` = 5 iterations, 5 queries. Default: `deep` |
| `max_search_iterations` | int 1–5 | No | Overrides the iteration limit from `depth` |
| `output_format` | `markdown` | No | Output format. Only `markdown` supported in v1 |

---

## Running Tests

```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app --cov-report=term-missing
```

Run a specific test file:
```bash
pytest tests/test_graph.py -v
```

### Test Structure

| File | What it tests |
|---|---|
| `tests/test_security.py` | Auth dependency — missing key, wrong key, valid key |
| `tests/test_jobs.py` | POST and GET job endpoints — happy path, validation, 429, 404 |
| `tests/test_stream.py` | SSE stream — event delivery, sentinel, keepalive, 404 |
| `tests/test_graph.py` | `resolve_max_iterations` and `_reflection_router` logic |
| `tests/test_services.py` | `chunk_text`, `embed_texts`, `_clean_html`, search fallback |

Tests never hit a real database, LLM, or search API — all external dependencies are mocked.

---

## Environment Variables

See `.env.example` for the full list. Minimum required:

```
API_KEY=your-secret-key
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/deep_research
OPENAI_API_KEY=sk-...
```

---

## Project Structure

```
app/
├── core/        # Config, DB engine, LLM factory, auth dependency
├── models/      # ORM models (DB schema) + Pydantic schemas (API contracts)
├── services/    # Search, scraper, embedder (stateless, reusable)
├── graph/       # LangGraph state machine + all nodes
│   └── nodes/   # planner, search_scrape, deduplication, reflection, report_generator
├── api/         # FastAPI routers (jobs, stream)
└── main.py      # App entrypoint + lifespan
migrations/      # Alembic versioned schema migrations
docs/            # PROJECT.md (full architecture + decisions)
tests/           # Test suite
```
