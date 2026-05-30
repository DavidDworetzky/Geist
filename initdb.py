import importlib

from dotenv import load_dotenv

from app.models.database.database import Base, DATABASE_BACKEND, Engine


load_dotenv()


def ensure_postgres_database() -> None:
    import os

    from psycopg2 import connect, sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    db_name = os.getenv("POSTGRES_DB", "geist")
    db_user = os.getenv("POSTGRES_USER", "geist")
    db_password = os.getenv("POSTGRES_PASSWORD", "geist")
    db_host = os.getenv("DB_HOST", "db")
    db_port = os.getenv("DB_PORT", "5432")

    conn = connect(
        dbname="postgres",
        user=db_user,
        host=db_host,
        port=db_port,
        password=db_password,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()

    if not exists:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"Database '{db_name}' created successfully")
    else:
        print(f"Database '{db_name}' already exists")

    cur.close()
    conn.close()


def main() -> None:
    if DATABASE_BACKEND != "sqlite":
        ensure_postgres_database()

    # Import all models to register them with Base.metadata.
    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=Engine)

    from scripts.insert_presets import main as insert_presets

    insert_presets(to_commit=True, overwrite=False)


if __name__ == "__main__":
    main()
