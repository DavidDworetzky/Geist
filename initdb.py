#Create database if it doesn't exist with psycopg2
from psycopg2 import connect, extensions, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.errors import DuplicateDatabase
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from scripts.insert_presets import main as insert_presets
import os

load_dotenv()

pwd = os.getenv("POSTGRES_PWD")
db_host = os.getenv("DB_HOST")

DB_NAME = "geist"
DB_USER = "postgres"
DEFAULT_DB_NAME = "postgres"
DB_HOST = "db" if not db_host else db_host
#Create database if it doesn't exist with psycopg2

conn = connect(
        dbname=DEFAULT_DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=5432,
        password=pwd,
    )
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()
try:
    cur.execute(sql.SQL("CREATE DATABASE {}".format(DB_NAME)))
except Exception as e:
    print(e)

cur.close()
conn.close()

#database imports
from app.models.database.database import Base
from app.models.database.database import Engine

#create models if they do not exist
Base.metadata.create_all(bind=Engine)

#now, run all other idempotent insertion scripts
insert_presets(to_commit=True)
