"""
app/main.py

WHAT this file does:
  - Creates the FastAPI application instance.
  - Registers all API routers.
  - Defines the lifespan context manager for startup and shutdown hooks.

LIFESPAN (startup / shutdown):
  - WHY lifespan over @app.on_event: FastAPI deprecated on_event in favour of
    the lifespan context manager. It is cleaner — startup and shutdown logic
    live together in one place.
  - Startup: nothing to initialise explicitly. SQLAlchemy engine and LLM client
    are lazy singletons (created on first use). Playwright browser is also lazy.
  - Shutdown: we explicitly close the Playwright browser process. Without this,
    the headless Chromium process would be orphaned when the app exits.

WHY no global exception handler for 500s:
  - FastAPI's default 500 handler returns a generic JSON error. That is sufficient
    for v1. Unhandled exceptions in background tasks are caught in _run_job_background
    and stored as error_message on the job record.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import jobs, stream
from app.services.scraper import close_browser


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    # Nothing to eagerly initialise — all clients are lazy singletons.
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    # Close the Playwright browser process cleanly.
    await close_browser()


app = FastAPI(
    title="Autonomous Deep Research Agent",
    version="1.0.0",
    description="Autonomous AI research service that searches, scrapes, deduplicates, and synthesises cited Markdown reports.",
    lifespan=lifespan,
)

app.include_router(jobs.router)
app.include_router(stream.router)


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Unauthenticated health check. Returns 200 if the app is running."""
    return {"status": "ok"}
