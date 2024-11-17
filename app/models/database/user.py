import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
import uuid

class User(Base):
    __tablename__ = "user"
    user_id = Column(Integer, primary_key=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    #hashed password
    password = Column(String)