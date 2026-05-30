# SQLite Storage Engine Plan

## Goal

Add an opt-in SQLite persistence implementation for the existing SQLAlchemy database layer while keeping PostgreSQL as the default runtime backend.

## Approach

1. Extend `app.models.database.database` so it can build either:
   - the current PostgreSQL engine from `POSTGRES_*` variables, or
   - a SQLite engine from `GEIST_DATABASE_BACKEND=sqlite` plus `SQLITE_DATABASE_PATH` or `SQLITE_DATABASE_URL`.
2. Preserve the current public API used throughout the codebase:
   - `Engine`
   - `SessionLocal`
   - `Session`
   - `Base`
   - `SQLALCHEMY_DATABASE_URL`
3. Make model declarations portable where they currently depend on PostgreSQL-specific types.
4. Update database initialization so SQLite creates its file-backed database and tables without requiring `psycopg2` setup.
5. Add unit tests that bind the application models to a temporary SQLite file and validate persistence across the existing model helper functions.

## Verification

1. Run focused SQLite unit tests locally.
2. Run the repository Python tests that do not require model downloads or external API credentials.
3. Start Docker with `docker compose up -d`, inspect backend logs for startup errors, and smoke test the frontend endpoint with `curl http://localhost:3000`.

## Configuration

PostgreSQL remains the default. To opt into SQLite:

```bash
GEIST_DATABASE_BACKEND=sqlite
SQLITE_DATABASE_PATH=/opt/geist/data/geist.sqlite3
```

or:

```bash
GEIST_DATABASE_BACKEND=sqlite
SQLITE_DATABASE_URL=sqlite:////opt/geist/data/geist.sqlite3
```
