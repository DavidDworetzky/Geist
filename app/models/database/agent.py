import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert

class Agent(Base):
    '''
    SqlAlchemy Class used to persist agent state.
    '''
    __tablename__ = "agent"
    agent_id = Column(Integer, primary_key = True)
    process_id = Column(String)
    world_context = Column(String)
    task_context = Column(String)
    execution_context = Column(String)
    