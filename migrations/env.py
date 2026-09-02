"""
migrations/env.py

WHY a custom async env.py:
  - The default Alembic env.py uses synchronous SQLAlchemy connections.
  - Our engine is async (asyncpg). Alembic requires a sync connection for
    migrations, so we use `engine.sync_engine` to get a sync handle from
    the async engine — the recommended pattern for async SQLAlchemy + Alembic.

WHY we import all ORM models here:
  - Alembic autogenerate works by comparing `Base.metadata` (all registered
    ORM models) against the live DB schema.
  - Models must be imported before `run_migrations_offline/online` is called
    so SQLAlchemy has registered them onto Base.metadata.
  - Importing `app.models.orm` here is the single place that ensures this.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.database import Base

# Import all ORM models so Alembic can detect them for autogenerate.
import app.models.orm  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """
    Run migrations against a live DB.
    Uses create_async_engine + run_sync to bridge async engine with Alembic's
    sync migration runner — the standard pattern for asyncpg + Alembic.
    """
    connectable = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,  # No pooling during migrations
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
