# Product Requirement Document (PRD)
## Autonomous Deep Research Agent Service

**Document Status:** Final Draft  
**Version:** 1.0.0  
**Target Architecture:** FastAPI | LangGraph | PostgreSQL (pgvector) | Playwright / Tavily  
**Primary AI Engines:** Claude 3.5 Sonnet / GPT-4o  

---

## 1. Executive Summary & Vision

### 1.1 Problem Statement
Standard search engines and raw LLM interfaces suffer from two critical limitations when conducting in-depth research:
1. **Single-pass limitation:** Search engines return query links, but require human manual extraction, aggregation, and cross-referencing.
2. **Hallucination & Recency issue:** Raw LLMs lack real-time web awareness, cannot verify facts across multiple sources, and fail to generate properly cited academic/business research papers.

### 1.2 Product Vision
The **Autonomous Deep Research Agent** is a backend service that acts as an autonomous AI analyst. Given a complex topic or question, the system breaks down the topic into focused sub-queries, navigates the live web, scrapes relevant pages, eliminates duplicate information using vector similarity, identifies gaps in its own knowledge base, iteratively executes follow-up searches, and synthesizes a fully cited markdown report—all while streaming real-time state logs to the client via Server-Sent Events (SSE).

---

## 2. Goals & Non-Goals

### 2.1 Core Goals
* **Autonomous Multi-Step Discovery:** Execute multi-turn search and scrape tasks without human intervention.
* **Vector-Based Deduplication:** Use vector embeddings and cosine similarity to prevent redundant source reading and token burn.
* **Self-Reflection Loop:** Evaluate gathered research against target goals to generate autonomous follow-up questions for knowledge gaps.
* **Deterministic Structured Output:** Produce a formatted Markdown paper complete with numbered inline citations mapped to verified URLs.
* **Real-Time Visibility:** Stream detailed step-by-step state progress (e.g., `Planning queries`, `Scraping domain X`, `Deduplicating vectors`) over SSE.

### 2.2 Non-Goals
* Human-in-the-loop manual content approval mid-research (v1 is fully autonomous once initiated).
* Web browsing behind auth-gated enterprise platforms (e.g., private corporate intranets).
* PDF/Doc document processing within the web search context (handled by separate IDP pipeline).

---

## 3. System Architecture & State Machine

### 3.1 High-Level Architecture
```
                         +-------------------------+
                         |      Client App         |
                         +------------+------------+
                                      |
                     POST /research   |  GET /stream (SSE)
                                      v
                         +------------+------------+
                         |    FastAPI Gateway      |
                         +------------+------------+
                                      |
                                      v
                         +------------+------------+
                         |  LangGraph Orchestrator |
                         +------------+------------+
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
    +--------+-------+       +--------+-------+       +--------+-------+
    | Tavily / DDG   |       | Scraper Engine |       | pgvector Store |
    | Search Tool    |       | (Playwright)   |       | (Deduplication)|
    +----------------+       +----------------+       +----------------+
```

### 3.2 State Machine Specification (LangGraph Workflow)

```
   [ START ]
       │
       ▼
[ Query Planner ] ──────────────► Decomposes topic into N sub-queries
       │
       ▼
[ Search & Scrape Node ] ───────► Executes parallel web queries & scrapes text
       │
       ▼
[ Deduplication Node ] ─────────► Embeds chunks; drops cosine similarity > 0.85
       │
       ▼
[ Reflection Node ] ────────────► Evaluates knowledge sufficiency
       │
       ├──── Knowledge Gap Exists AND Loop Count < Max ───► [ Sub-Query Generator ]
       │                                                            │
       │                                                            └─► (Loop back to Search)
       │
       └──── Knowledge Sufficient OR Loop Count >= Max
               │
               ▼
      [ Report Generator ] ─────► Formats markdown, maps citations, finalize output
               │
               ▼
            [ END ]
```

---

## 4. Functional Requirements & Specifications

### 4.1 Topic Ingestion & Decomposition
* **Input Parameters:** Topic string, depth setting (`fast`: max 2 iterations; `deep`: max 5 iterations), max sources threshold.
* **Query Decomposition:** System prompt utilizes LLM structured output to break down primary research topics into 3–5 targeted search queries.

### 4.2 Autonomous Web Search & Scraping
* **Search Integration:** Primary integration via Tavily AI Search API with fallback to DuckDuckGo Python API.
* **Web Scraping:** Asynchronous HTML parsing via Playwright or BeautifulSoup. Strips scripts, ads, and navigation boilerplate to retain core article text.

### 4.3 Vector Embedding & Content Deduplication
* **Embedding Model:** `text-embedding-3-small` or `bge-small-en-v1.5`.
* **Storage & Indexing:** Temporary session namespace in `pgvector` or Qdrant.
* **Deduplication Logic:** Calculate cosine similarity between newly scraped text chunks ($C_{new}$) and already indexed chunks ($C_{existing}$). If $	ext{similarity}(C_{new}, C_{existing}) \ge 0.85$, discard $C_{new}$ to conserve context space.

### 4.4 Self-Reflection & Iterative Research Loop
* **Knowledge Evaluation:** The reflection node compares current accumulated facts against original topic scope.
* **Gap Identification:** Generates follow-up sub-questions specifically targeting missing information (e.g., missing metrics, conflicting facts, missing recent developments).
* **Recursion Guard:** Strict guardrail enforcing maximum iteration count ($N \le 5$) regardless of perceived gaps.

### 4.5 Report Synthesis & Citation Engine
* **Markdown Formatting:** Executive Summary, Methodology, Deep Dive Sections, Limitations, and References.
* **Citation Mapping:** Inline markers `[1]`, `[2]` linking directly to verified, scraped URLs in the References section.

---

## 5. API Interface & Data Contracts

### 5.1 POST /api/v1/research/jobs
Triggers a new deep research execution.

**Request Payload:**
```json
{
  "topic": "Impact of Rust on Backend System Performance vs Go in 2026",
  "depth": "deep",
  "max_search_iterations": 4,
  "output_format": "markdown"
}
```

**Response Payload (202 Accepted):**
```json
{
  "job_id": "8f3b2a9d-5e1c-4b3a-9f8e-2d1c0b9a8f7e",
  "status": "QUEUED",
  "created_at": "2026-08-21T00:00:00Z",
  "sse_stream_url": "/api/v1/research/jobs/8f3b2a9d-5e1c-4b3a-9f8e-2d1c0b9a8f7e/stream"
}
```

### 5.2 GET /api/v1/research/jobs/{job_id}/stream
Server-Sent Events (SSE) streaming live execution progress.

**Event Stream Format:**
```
event: state_change
data: {"step": "PLANNING", "message": "Decomposed topic into 4 sub-queries", "iteration": 1}

event: search_executed
data: {"step": "SEARCHING", "query": "Rust vs Go memory footprint benchmarks 2025 2026", "urls_found": 5}

event: deduplication_complete
data: {"step": "DEDUPLICATING", "raw_chunks": 42, "unique_chunks_retained": 18}

event: reflection
data: {"step": "REFLECTING", "gap_identified": "Lack of production garbage collection latency metrics", "decision": "CONTINUE"}

event: complete
data: {"step": "COMPLETED", "report_url": "/api/v1/research/jobs/8f3b2a9d-5e1c-4b3a-9f8e-2d1c0b9a8f7e"}
```

---

## 6. Database Schema (PostgreSQL + pgvector)

```sql
-- Job Tracking Table
CREATE TABLE research_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,
    depth VARCHAR(20) NOT NULL DEFAULT 'deep',
    status VARCHAR(30) NOT NULL DEFAULT 'QUEUED',
    iteration_count INT DEFAULT 0,
    report_markdown TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector Store for Sources & Deduplication
CREATE TABLE research_source_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES research_jobs(job_id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content_chunk TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_source_chunks_vector ON research_source_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## 7. Edge Cases & Resilience Engineering

| Edge Case / Failure | Cause | Mitigation Architecture |
| :--- | :--- | :--- |
| **Scraper Blocking / 403 / Captcha** | Target site blocks bot access | Fallback to Tavily extracted snippet body or headless browser proxy rotation. |
| **Infinite Loop / Hallucinated Loops** | Reflection node continually finds minor knowledge gaps | Enforce hard code cap (`iteration >= max_iterations`), forcing completion. |
| **Context Window Overflow** | Accumulating dozens of web pages | Compress research state by summarizing historical chunks per iteration before final synthesis. |
| **Dead SSE Connections** | Client disconnects mid-stream | Research runs asynchronously in background worker; client can fetch final result via REST endpoint at any time. |

---

## 8. Development Implementation Roadmap

* **Phase 1 (Days 1–3):** Foundation & LangGraph Setup
  * Implement base FastAPI setup and define state structures.
  * Construct primary LangGraph flow: Planner -> Search -> Synthesis.
* **Phase 2 (Days 4–6):** Scraper & Deduplication Engine
  * Integrate Tavily / Playwright tools.
  * Connect `pgvector` embedding pipeline with cosine similarity filtering ($\ge 0.85$).
* **Phase 3 (Days 7–9):** Reflection Loop & Citation Logic
  * Implement reflection node gap analysis.
  * Build deterministic citation mapper ensuring every statement maps to scraped URLs.
* **Phase 4 (Days 10–12):** SSE Progress Streaming & Resilience Testing
  * Implement FastAPI SSE handler.
  * Add circuit breakers, retry logic, and handle rate-limiting scenarios.
