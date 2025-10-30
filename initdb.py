from psycopg2 import connect, extensions, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.errors import DuplicateDatabase
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os

load_dotenv()

# Use environment variables set in docker-compose.yml
DB_NAME = os.getenv("POSTGRES_DB", "geist")
DB_USER = os.getenv("POSTGRES_USER", "geist")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "geist")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

# Connect to the default 'postgres' database first
conn = connect(
    dbname="postgres",
    user=DB_USER,
    host=DB_HOST,
    port=DB_PORT,
    password=DB_PASSWORD,
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

# Check if database exists
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
exists = cur.fetchone()

if not exists:
    try:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' created successfully")
    except Exception as e:
        print(f"An error occurred while creating the database: {e}")
else:
    print(f"Database '{DB_NAME}' already exists")

cur.close()
conn.close()

# Now connect to the new database to create tables
conn = connect(
    dbname=DB_NAME,
    user=DB_USER,
    host=DB_HOST,
    port=DB_PORT,
    password=DB_PASSWORD,
)

# Rest of your code...
#database imports
from app.models.database.database import Base
from app.models.database.database import Engine
# Import all models to register them with Base.metadata
import app.models.database

#create models if they do not exist
Base.metadata.create_all(bind=Engine)

#now, run all other idempotent insertion scripts
from scripts.insert_presets import main as insert_presets
# Add the 'overwrite' argument here. You may want to set this to False if you don't want to overwrite existing data
insert_presets(to_commit=True, overwrite=False)

conn.close()