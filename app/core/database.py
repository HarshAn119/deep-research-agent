"""
app/core/database.py

WHY async SQLAlchemy + asyncpg:
  - FastAPI is fully async. Using a sync DB driver would block the event loop
    during every query, defeating the purpose of async request handling.
  - asyncpg is the fastest async PostgreSQL driver available for Python.

HOW:
  - `engine` is created once at module load using the DATABASE_URL from settings.
  - `AsyncSessionLocal` is a session factory — each request gets its own session.
  - `get_db` is a FastAPI dependency that yields a session and guarantees cleanup.
  - `Base` is the declarative base all ORM models inherit from.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    # ASSUMPTION: pool_size=10 matches MAX_CONCURRENT_JOBS default.
    # Each running job may hold one DB connection for its duration.
    pool_size=10,
    max_overflow=5,
    echo=settings.app_env == "development",  # SQL logging in dev only
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,  # Prevents lazy-load errors after commit in async context
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request, always closes it."""
    async with AsyncSessionLocal() as session:
        yield session
