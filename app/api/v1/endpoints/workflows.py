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
from sqlalchemy.orm import Session, selectinload
DEFAULT_USER_ID = 1
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
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Create a new workflow."""
    db_workflow = Workflow(
        name=workflow.name,
        user_id=DEFAULT_USER_ID  # Using default user_id for now
    )
    
    if workflow.steps:
        db_workflow.steps = [
            WorkflowStep(**step.dict())
            for step in workflow.steps
        ]
    
    return create_workflow(db_workflow)

@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    db: Session = Depends(get_db)
) -> List[WorkflowResponse]:
    """List all workflows."""
    return get_workflows_for_user(DEFAULT_USER_ID)  # Using default user_id for now

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Get a specific workflow by ID."""
    workflow = db.query(Workflow).options(selectinload(Workflow.steps)).filter(Workflow.workflow_id == workflow_id).first()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    return workflow

@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_existing_workflow(
    workflow_id: int,
    workflow_update: WorkflowUpdate,
    db: Session = Depends(get_db)
) -> WorkflowResponse:
    """Update an existing workflow."""
    initial_check_workflow = get_workflow_by_id(workflow_id=workflow_id)

    if not initial_check_workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )

    try:
        updated_db_workflow = update_workflow(workflow_id=workflow_id, workflow_data=workflow_update)
        if updated_db_workflow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID {workflow_id} not found during update process."
            )
        return updated_db_workflow
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: int,
    db: Session = Depends(get_db)
) -> None:
    """Delete a workflow."""
    workflow = get_workflow_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    with SessionLocal() as session:
        session.delete(workflow)
        session.commit() 