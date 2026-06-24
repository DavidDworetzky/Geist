# Injectable SQLAlchemy Provider Plan

## Goal

Add a provider-neutral SQLAlchemy configuration layer that can select PostgreSQL or SQLite at runtime while keeping PostgreSQL as the default provider.

## Approach

1. Add an immutable `DatabaseConfig` and provider registry outside the process-wide session module.
2. Implement PostgreSQL and SQLite as registered providers responsible for:
   - building their default connection URL,
   - creating their SQLAlchemy engine, and
   - performing provider-specific database initialization.
3. Load configuration from environment variables only as the default adapter. Tests and application composition can inject a `DatabaseConfig` directly without mutating process environment.
4. Preserve the current public API used throughout the codebase:
   - `Engine`
   - `SessionLocal`
   - `Session`
   - `Base`
   - `SQLALCHEMY_DATABASE_URL`
5. Make model declarations portable where they currently depend on PostgreSQL-specific types.
6. Update database initialization to delegate setup to the selected provider.
7. Add unit tests for provider selection, direct configuration injection, and SQLite persistence across existing model helper functions.
8. Exercise the shared agent lifecycle against SQLite for both `LocalAgent` and
   `OnlineAgent`, with inference and HTTP boundaries mocked:
   - initialization and state inspection,
   - chat completion persistence,
   - context persistence during phase out, and
   - phase in without subprocess creation.

## Verification

1. Run focused SQLite unit tests locally.
2. Run the repository Python tests that do not require model downloads or external API credentials.
3. Start Docker with `docker compose up -d`, inspect backend logs for startup errors, and smoke test the frontend endpoint with `curl http://localhost:3000`.

## Configuration

PostgreSQL remains the default. Provider selection uses:

```bash
GEIST_DATABASE_PROVIDER=sqlite
SQLITE_DATABASE_PATH=/opt/geist/data/geist.sqlite3
```

or:

```bash
GEIST_DATABASE_PROVIDER=sqlite
SQLITE_DATABASE_URL=sqlite:////opt/geist/data/geist.sqlite3
```

`GEIST_DATABASE_BACKEND` and `DB_BACKEND` remain supported as compatibility aliases. Application code and tests can bypass environment loading entirely:

```python
DatabaseConfig(
    provider="sqlite",
    database_url="sqlite:////tmp/geist.sqlite3",
)
```
