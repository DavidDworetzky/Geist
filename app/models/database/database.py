from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

pwd = os.getenv("POSTGRES_PWD")
db = os.getenv("POSTGRES_DB")

if db is None:
    db = "platz"

# We take a default connection to start. 
# We should pull this from env / secrets when securing our server.
SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{pwd}@localhost/{db}"

#instantiate database engine and session
Engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Session = SessionLocal()

Base = declarative_base()
