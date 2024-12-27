from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# Use environment variables set in docker-compose.yml
DB_NAME = os.getenv("POSTGRES_DB", "geist")
DB_USER = os.getenv("POSTGRES_USER", "geist")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "geist")
DB_HOST = os.getenv("DB_HOST", "geist")
DB_PORT = os.getenv("DB_PORT", "5432")

# Construct the database URL
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Instantiate database engine and session
Engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Session = SessionLocal()

Base = declarative_base()