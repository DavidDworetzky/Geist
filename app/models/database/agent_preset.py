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
    #interactive_only - is not an independent agent.
    interactive_only = Column(Boolean)
    # optional processing settings
    process_world = Column(Boolean)
    #restriction relationships
    restrictions = relationship("Restriction", back_populates="agent_preset")
    create_date = Column(DateTime)
    update_date = Column(DateTime)

    @classmethod
    def upsert_agent_preset(cls, name, version, description, max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty, tags, working_context_length, long_term_context_length, agent_type, prompt, interactive_only, process_world):
        """
        Upserts an AgentPreset entity into the database.
        """
        session = Session
        try:
            # Query for existing AgentPreset
            existing_preset = session.query(cls).filter_by(name=name, version=version).first()
            
            # If exists, update
            if existing_preset:
                existing_preset.description = description
                existing_preset.max_tokens = max_tokens
                existing_preset.n = n
                existing_preset.temperature = temperature
                existing_preset.top_p = top_p
                existing_preset.frequency_penalty = frequency_penalty
                existing_preset.presence_penalty = presence_penalty
                existing_preset.tags = tags
                existing_preset.working_context_length = working_context_length
                existing_preset.long_term_context_length = long_term_context_length
                existing_preset.agent_type = agent_type
                existing_preset.prompt = prompt
                existing_preset.interactive_only = interactive_only
                existing_preset.process_world = process_world
            else:
                # If doesn't exist, create new
                new_preset = cls(
                    name=name,
                    version=version,
                    description=description,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    tags=tags,
                    working_context_length=working_context_length,
                    long_term_context_length=long_term_context_length,
                    agent_type=agent_type,
                    prompt=prompt,
                    interactive_only=interactive_only
                    process_world=process_world
                )
                session.add(new_preset)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

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
    create_date = Column(DateTime)
    update_date = Column(DateTime)

