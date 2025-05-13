import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass

import uuid

class Workflow(Base):
    __tablename__ = "worklfow"
    workflow_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)


class WorkflowStep(Base):
    __tablename__ = "workflow_step"
    step_id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(Integer, ForeignKey("workflow.workflow_id"))
    step_name = Column(String)
    step_description = Column(String)
    step_type = Column(String)
    step_status = Column(String)
    #for workflow editor location
    display_x = Column(Integer)
    display_y = Column(Integer)
    #command string for workflow step
    commmmand_str = Column(String)