#commands
`docker compose up` to run the docker container
`make build` to build the solution
`make run MLX_BACKEND=1` to run the solution with MLX_BACKEND instead of 
`make empty` to run an empty container to install dependencies

`docker exec backend /bin/bash` to enter the backend container
`pip install PACKAGE` to install dependencies
When installing packages, `conda env export >> linux_environment.yml` after installing to freeze installs. 


#SQLAlchemy
When adding classes to sql alchemy, take the following example
EXAMPLE: 
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert

class AgentPreset(Base):
    """
    Class used to represent a preset for an agent.
    """
    __tablename__ = 'agent_preset'
    agent_preset_id = Column(Integer, primary_key=True, autoincrement=True)
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
    #interactive_only - is not an independent agent.
    interactive_only = Column(Boolean)
    # optional processing settings
    process_world = Column(Boolean)
    #restriction relationships
    restrictions = relationship("Restriction", back_populates="agent_preset")
    create_date = Column(DateTime)
    update_date = Column(DateTime)

classes are stored in app >> models >> database. 

#Adding Models
Models should be added to scripts/copy_weights.py as well as the associated agent implementation inheriting from agents/base_agent. (GPT4 Agent, llama_agent, etc.)



