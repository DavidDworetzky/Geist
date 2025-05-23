import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import uuid

class Workflow(Base):
    """Model representing a workflow in the system.
    
    Attributes:
        workflow_id (int): Primary key, auto-incrementing
        user_id (int): Foreign key to geist_user table
        name (str): Name of the workflow
        steps (List[WorkflowStep]): List of steps in the workflow
    """
    __tablename__ = "workflow"
    workflow_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"))
    name = Column(String)
    steps = relationship("WorkflowStep", back_populates="workflow", cascade="all, delete-orphan")

class WorkflowStep(Base):
    """Model representing a step within a workflow.
    
    Attributes:
        step_id (int): Primary key, auto-incrementing
        workflow_id (int): Foreign key to workflow table
        step_name (str): Name of the step
        step_description (str): Description of the step
        step_status (str): Current status of the step
        display_x (int): X coordinate for workflow editor
        display_y (int): Y coordinate for workflow editor
        commmmand_str (str): Command string for workflow step
        step_type (str): Type of the step
        workflow (Workflow): Relationship to parent workflow
    """
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
    workflow = relationship("Workflow", back_populates="steps")

class WorkflowStepType(Enum):
    """Enum representing different types of workflow steps."""
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

def get_workflow_by_id(workflow_id: int) -> Optional[Workflow]:
    """Get a workflow by its ID.
    
    Args:
        workflow_id (int): The ID of the workflow to retrieve
        
    Returns:
        Optional[Workflow]: The workflow if found, None otherwise
    """
    with SessionLocal() as session:
        return session.query(Workflow).filter_by(workflow_id=workflow_id).first()
    
def get_workflows_for_user(user_id: int) -> List[Workflow]:
    """Get all workflows for a specific user.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        List[Workflow]: List of workflows belonging to the user
    """
    with SessionLocal() as session:
        return session.query(Workflow).filter_by(user_id=user_id).all()
    
def create_workflow(workflow: Workflow) -> Workflow:
    """Create a new workflow.
    
    Args:
        workflow (Workflow): The workflow to create
        
    Returns:
        Workflow: The created workflow
    """
    with SessionLocal() as session:
        session.add(workflow)
        session.commit()
        return workflow
    
def update_workflow(workflow: Workflow) -> Workflow:
    """Update an existing workflow.
    
    Args:
        workflow (Workflow): The workflow with updated data
        
    Returns:
        Workflow: The updated workflow
        
    Raises:
        ValueError: If the workflow is not found
    """
    with SessionLocal() as session:
        workflow_in_db = session.query(Workflow).filter_by(workflow_id=workflow.workflow_id).first()
        if workflow_in_db is None:
            raise ValueError("Workflow not found")
        
        workflow_in_db.name = workflow.name
        workflow_in_db.steps = workflow.steps
        session.commit()
        session.refresh(workflow_in_db)
        return workflow_in_db