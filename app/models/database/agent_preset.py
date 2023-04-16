import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from app.models.database.database import Base

class AgentPreset(Base):
    """
    Class used to represent a preset for an agent.
    """
    __tablename__ = 'agent_preset'
    agent_preset_id = Column(Integer, primary_key=True)
    #language model presets
    name = Column(String)
    version = Column(String)
    description = Column(String)
    max_tokens = Column(Integer)
    n = Column(Integer)
    temperature = Column(Integer)
    top_p = Column(Integer)
    frequency_penalty = Column(Integer)
    presence_penalty = Column(Integer)
    tags = Column(String)
    #memory presets
    working_context_length = Column(Integer)
    long_term_context_length = Column(Integer)
    agent_type = Column(String)
    #prompt presets
    prompt = Column(String)
    #restriction relationships
    restrictions = relationship("Restriction", back_populates="agent_preset")


