import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime, Text, JSON
from sqlalchemy.orm import relationship, Session, selectinload
from app.models.database.database import Base
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TYPE_CHECKING, Dict, Any

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
    # Input type - triggers that start workflows
    TRIGGER = "trigger"
    # Processing types
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
    # Output type - adapters that send data to external systems
    ADAPTER = "adapter"

class WorkflowRunStatus(Enum):
    """Enum representing different statuses of workflow runs."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowRun(Base):
    """Model representing a workflow execution run.
    
    Attributes:
        run_id (int): Primary key, auto-incrementing
        workflow_id (int): Foreign key to workflow table
        user_id (int): Foreign key to geist_user table
        status (str): Current status of the run
        started_at (datetime): When the run started
        completed_at (datetime): When the run completed
        error_message (str): Error message if run failed
        input_data (dict): Input data for the workflow run
        output_data (dict): Output data from the workflow run
        workflow (Workflow): Relationship to parent workflow
        step_results (List[WorkflowStepResult]): Results from each step
    """
    __tablename__ = "workflow_run"
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(Integer, ForeignKey("workflow.workflow_id"))
    user_id = Column(Integer, ForeignKey("geist_user.user_id"))
    status = Column(String, default=WorkflowRunStatus.RUNNING.value)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    input_data = Column(JSON)
    output_data = Column(JSON)
    workflow = relationship("Workflow")
    step_results = relationship("WorkflowStepResult", back_populates="run", cascade="all, delete-orphan")

class WorkflowStepResult(Base):
    """Model representing the result of a workflow step execution.
    
    Attributes:
        result_id (int): Primary key, auto-incrementing
        run_id (int): Foreign key to workflow_run table
        step_id (int): Foreign key to workflow_step table
        status (str): Status of the step execution
        started_at (datetime): When the step started
        completed_at (datetime): When the step completed
        input_data (dict): Input data for the step
        output_data (dict): Output data from the step
        error_message (str): Error message if step failed
        run (WorkflowRun): Relationship to parent run
        step (WorkflowStep): Relationship to workflow step
    """
    __tablename__ = "workflow_step_result"
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("workflow_run.run_id"))
    step_id = Column(Integer, ForeignKey("workflow_step.step_id"))
    status = Column(String, default=WorkflowRunStatus.RUNNING.value)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    run = relationship("WorkflowRun", back_populates="step_results")
    step = relationship("WorkflowStep")

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
        
        # Eagerly load the steps relationship before closing the session
        workflow = session.query(Workflow).options(
            selectinload(Workflow.steps)
        ).filter(Workflow.workflow_id == workflow.workflow_id).one()
        
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
        
        # Eagerly load the steps relationship before closing the session
        workflow_in_db = session.query(Workflow).options(
            selectinload(Workflow.steps)
        ).filter(Workflow.workflow_id == workflow_id).one()
        
        return workflow_in_db

def create_workflow_run(workflow_id: int, user_id: int, input_data: Dict[str, Any] = None) -> WorkflowRun:
    """Create a new workflow run.
    
    Args:
        workflow_id (int): The ID of the workflow to run
        user_id (int): The ID of the user running the workflow
        input_data (dict): Input data for the workflow run
        
    Returns:
        WorkflowRun: The created workflow run
    """
    with SessionLocal() as session:
        run = WorkflowRun(
            workflow_id=workflow_id,
            user_id=user_id,
            input_data=input_data or {},
            status=WorkflowRunStatus.RUNNING.value
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

def update_workflow_run_status(run_id: int, status: WorkflowRunStatus, error_message: str = None, output_data: Dict[str, Any] = None) -> Optional[WorkflowRun]:
    """Update the status of a workflow run.
    
    Args:
        run_id (int): The ID of the workflow run to update
        status (WorkflowRunStatus): The new status
        error_message (str): Error message if the run failed
        output_data (dict): Output data from the workflow run
        
    Returns:
        Optional[WorkflowRun]: The updated workflow run, or None if not found
    """
    with SessionLocal() as session:
        run = session.query(WorkflowRun).filter_by(run_id=run_id).first()
        if run:
            run.status = status.value
            run.completed_at = datetime.datetime.utcnow()
            if error_message:
                run.error_message = error_message
            if output_data:
                run.output_data = output_data
            session.commit()
            session.refresh(run)
            return run
        return None

def create_workflow_step_result(run_id: int, step_id: int, input_data: Dict[str, Any] = None) -> WorkflowStepResult:
    """Create a new workflow step result.
    
    Args:
        run_id (int): The ID of the workflow run
        step_id (int): The ID of the workflow step
        input_data (dict): Input data for the step
        
    Returns:
        WorkflowStepResult: The created step result
    """
    with SessionLocal() as session:
        result = WorkflowStepResult(
            run_id=run_id,
            step_id=step_id,
            input_data=input_data or {},
            status=WorkflowRunStatus.RUNNING.value
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return result

def update_workflow_step_result(result_id: int, status: WorkflowRunStatus, output_data: Dict[str, Any] = None, error_message: str = None) -> Optional[WorkflowStepResult]:
    """Update a workflow step result.
    
    Args:
        result_id (int): The ID of the step result to update
        status (WorkflowRunStatus): The new status
        output_data (dict): Output data from the step
        error_message (str): Error message if the step failed
        
    Returns:
        Optional[WorkflowStepResult]: The updated step result, or None if not found
    """
    with SessionLocal() as session:
        result = session.query(WorkflowStepResult).filter_by(result_id=result_id).first()
        if result:
            result.status = status.value
            result.completed_at = datetime.datetime.utcnow()
            if output_data:
                result.output_data = output_data
            if error_message:
                result.error_message = error_message
            session.commit()
            session.refresh(result)
            return result
        return None