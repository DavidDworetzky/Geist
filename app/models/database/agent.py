import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session, SessionLocal
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
import uuid


class Agent(Base):
    '''
    SqlAlchemy Class used to persist agent state.
    '''
    __tablename__ = "agent"
    agent_id = Column(Integer, primary_key = True)
    agent_identifier = Column(String, default=lambda: str(uuid.uuid4()))
    process_id = Column(String)
    world_context = Column(String)
    task_context = Column(String)
    execution_context = Column(String)


    @classmethod
    def get_agent_by_id(cls, agent_id):
        """
        Queries the database for an agent with the given agent_id.
        
        :param agent_id: The ID of the agent to retrieve.
        :return: The queried agent object if found, otherwise None.
        """
        session = SessionLocal()
        try:
            agent = session.query(cls).filter_by(agent_id=agent_id).first()
            return agent
        except Exception as e:
            session.rollback()
            print(f"Failed to query agent by id {agent_id}: {e}")
            return None
        finally:
            session.close()

    