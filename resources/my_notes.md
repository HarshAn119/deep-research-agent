- Pydantic is a Python library used to validate and structure data.
It means you create a config.py file that uses Pydantic to manage your app's configuration.
Pydantic validates them when the app starts and can read/override them from environment variables.
Think: config.py ≈ one typed, validated place for all your process.env.* values

- FastAPI → Python web framework, similar to Express.js in MERN.
- SQLAlchemy → Python library for working with SQL databases, similar to an ORM like Mongoose, but for databases like PostgreSQL/MySQL.
- asyncpg → A PostgreSQL driver that lets Python communicate with PostgreSQL asynchronously.
- Async SQLAlchemy → SQLAlchemy using async database operations, so FastAPI can handle other requests while waiting for the database.


- pyproject.toml - single dependency manifest. (python ka package.json)

- app/core/config.py - typed `BaseSettings` singleton. Fails at startup if a required variable is missing or invalid. All tuning constants (CHUNK_SIZE_TOKENS, DEDUP_SIMILARITY_THRESHOLD, etc.) live here — not scattered across the codebase.

- app/core/database.py - async SQLAlchemy engine + FastAPI dependency, Python library for working with SQL databases, similar to an ORM like Mongoose, but for databases like PostgreSQL/MySQL.
 Async SQLAlchemy → SQLAlchemy using async database operations, so FastAPI can handle other requests while waiting for the database.

- app/models/orm.py - ORM models are the schema source of truth. Alembic reads from these. JobStatus and DepthPreset are Python enums for type safety.

- app/models/schemas.py - Pydantic request/response contracts, fully decoupled from ORM. max_search_iterations is clamped to [1, 5] here before it ever reaches the graph. All SSE event shapes are typed here too.

- app/graph/state.py - The `AgentState` TypeDict that flows through every LangGraph node. `sse_queue` is injected at runtime, never persisted. `reflection_decision` drives the conditional edge.

- migrations - Alembic setup with an async-compatible env.py and a hand-written initial migration that creates the pgvector extension, enums, both tables, and the ivfflat index.