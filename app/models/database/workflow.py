import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass
from enum import Enum
from typing import List

import uuid

class Workflow(Base):
    __tablename__ = "worklfow"
    workflow_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"))
    name = Column(String)
    steps = relationship("WorkflowStep", back_populates="workflow")

class WorkflowStep(Base):
    __tablename__ = "workflow_step"
    step_id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(Integer, ForeignKey("workflow.workflow_id"))
    step_name = Column(String)
    step_description = Column(String)
    step_status = Column(String)
    #for workflow editor location
    display_x = Column(Integer)
    display_y = Column(Integer)
    #command string for workflow step
    commmmand_str = Column(String)
    #step type
    step_type = Column(String)

class WorkflowStepType(Enum):
    MAP = "map"
    FILTER = "filter"
    REDUCE = "reduce"
    EXPAND = "expand"
    #execute custom code, can call adapters
    CUSTOM = "custom"
    #transform from input to output with an LLM prompt
    LLM = "llm"
    #agent flow, can call an agent flow step
    AGENT = "agent"

def get_workflow_by_id(workflow_id: int) ->  List[WorkflowStep]:
    with SessionLocal() as session:
        workflow = session.query(Workflow).filter_by(workflow_id=workflow_id).first()
        return workflow.steps
    
def get_workflows_for_user(user_id: int) -> List[Workflow]:
    with SessionLocal() as session:
        workflows = session.query(Workflow).filter_by(user_id=user_id).all()
        return workflows
    
def create_workflow(workflow: Workflow) -> Workflow:
    with SessionLocal() as session:
        session.add(workflow)
        session.commit()
        return workflow
    
def update_workflow(workflow: Workflow) -> Workflow:
    with SessionLocal() as session:
        #find the workflow in the database
        workflow_in_db = session.query(Workflow).filter_by(workflow_id=workflow.workflow_id).first()
        if workflow_in_db is None:
            raise ValueError("Workflow not found")
        #update the workflow
        workflow_in_db.name = workflow.name
        workflow_in_db.steps = workflow.steps
        session.commit()