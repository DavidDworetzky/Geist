import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session, selectinload
from app.models.database.database import Base
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

import uuid

if TYPE_CHECKING:
    from app.schemas.workflow import WorkflowUpdate

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
        command_str (str): Command string for workflow step
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
    command_str = Column(String)
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
        return session.query(Workflow).options(selectinload(Workflow.steps)).filter_by(workflow_id=workflow_id).first()
    
def get_workflows_for_user(user_id: int, db: Session) -> List[Workflow]:
    """Get all workflows for a specific user.
    
    Args:
        user_id (int): The ID of the user
        db (Session): The database session.
        
    Returns:
        List[Workflow]: List of workflows belonging to the user
    """
    return db.query(Workflow).options(selectinload(Workflow.steps)).filter_by(user_id=user_id).all()
    
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
    
def update_workflow(workflow_id: int, workflow_data: 'WorkflowUpdate') -> Optional[Workflow]:
    """Update an existing workflow.
    
    Args:
        workflow_id (int): The ID of the workflow to update.
        workflow_data (WorkflowUpdate): The Pydantic schema containing updated data for the workflow.
        
    Returns:
        Optional[Workflow]: The updated workflow, or None if the workflow was not found.
        
    Raises:
        ValueError: If the workflow is not found.
    """
    with SessionLocal() as session:
        workflow_in_db = session.query(Workflow).filter_by(workflow_id=workflow_id).first()
        
        if workflow_in_db is None:
            raise ValueError(f"Workflow with ID {workflow_id} not found for update.")

        # Update the workflow's name if provided
        if workflow_data.name is not None:
            workflow_in_db.name = workflow_data.name

        # Update steps if provided in the payload
        if workflow_data.steps is not None:
            workflow_in_db.steps.clear() 
            session.flush() 

            new_orm_steps = []
            # No need to check if workflow_data.steps is empty list, loop will handle it.
            for step_payload in workflow_data.steps: 
                new_step = WorkflowStep(
                    step_name=step_payload.step_name,
                    step_description=step_payload.step_description,
                    step_status=step_payload.step_status,
                    display_x=step_payload.display_x,
                    display_y=step_payload.display_y,
                    command_str=step_payload.command_str, 
                    step_type=step_payload.step_type.value
                )
                new_orm_steps.append(new_step)
            
            workflow_in_db.steps = new_orm_steps
        # If workflow_data.steps is None, existing steps are not modified.

        session.add(workflow_in_db) 
        session.commit()
        session.refresh(workflow_in_db) 
        return workflow_in_db