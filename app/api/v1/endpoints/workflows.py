from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.schemas.workflow import (
    WorkflowCreate,
    WorkflowResponse,
    WorkflowUpdate,
    WorkflowStepCreate,
    WorkflowStepResponse
)
from app.models.database.workflow import Workflow, WorkflowStep, get_workflow_by_id, get_workflows_for_user, create_workflow, update_workflow
from app.models.database.database import SessionLocal
from sqlalchemy.orm import Session
from app.api.deps import get_current_user

router = APIRouter()

def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_new_workflow(
    workflow: WorkflowCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Create a new workflow.
    
    Args:
        workflow: The workflow data to create
        current_user: The current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: The created workflow
    """
    db_workflow = Workflow(
        name=workflow.name,
        user_id=current_user["user_id"]
    )
    
    if workflow.steps:
        db_workflow.steps = [
            WorkflowStep(**step.dict())
            for step in workflow.steps
        ]
    
    return create_workflow(db_workflow)

@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[WorkflowResponse]:
    """List all workflows for the current user.
    
    Args:
        current_user: The current authenticated user
        db: Database session
        
    Returns:
        List[WorkflowResponse]: List of workflows
    """
    return get_workflows_for_user(current_user["user_id"])

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Get a specific workflow by ID.
    
    Args:
        workflow_id: The ID of the workflow to retrieve
        current_user: The current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: The requested workflow
        
    Raises:
        HTTPException: If workflow not found or user doesn't have access
    """
    workflow = get_workflow_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    if workflow.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this workflow"
        )
    
    return workflow

@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_existing_workflow(
    workflow_id: int,
    workflow_update: WorkflowUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Update an existing workflow.
    
    Args:
        workflow_id: The ID of the workflow to update
        workflow_update: The updated workflow data
        current_user: The current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: The updated workflow
        
    Raises:
        HTTPException: If workflow not found or user doesn't have access
    """
    existing_workflow = get_workflow_by_id(workflow_id)
    if not existing_workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    if existing_workflow.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this workflow"
        )
    
    if workflow_update.name is not None:
        existing_workflow.name = workflow_update.name
    
    if workflow_update.steps is not None:
        # Clear existing steps
        existing_workflow.steps = []
        # Add new steps
        existing_workflow.steps = [
            WorkflowStep(**step.dict())
            for step in workflow_update.steps
        ]
    
    return update_workflow(existing_workflow)

@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> None:
    """Delete a workflow.
    
    Args:
        workflow_id: The ID of the workflow to delete
        current_user: The current authenticated user
        db: Database session
        
    Raises:
        HTTPException: If workflow not found or user doesn't have access
    """
    workflow = get_workflow_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    if workflow.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this workflow"
        )
    
    with SessionLocal() as session:
        session.delete(workflow)
        session.commit() 