import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, ARRAY, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from app.models.database.database import Base

class Restriction(Base):
    """
    Class used to represent a restriction for an agent.
    """
    __tablename__ = 'restriction'
    restriction_id = Column(Integer, primary_key=True)
    agent_preset_id = Column(Integer, ForeignKey('agent_preset.agent_preset_id'))
    agent_preset = relationship("AgentPreset", back_populates="restrictions")
    #restriction fields
    name = Column(String)
    rate = Column(Integer)
    period_hours = Column(Integer)
    spending_limit = Column(Integer)
    restriction_type = Column(String)
    #allowed plugins and methods that an agent can use
    allowed_plugins = Column(ARRAY(String))
    allowed_methods = Column(ARRAY(String))
