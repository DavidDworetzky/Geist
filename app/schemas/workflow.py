from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.database.workflow import WorkflowStepType

class WorkflowStepBase(BaseModel):
    """Base schema for workflow step data."""
    step_name: str = Field(..., description="Name of the workflow step")
    step_description: Optional[str] = Field(None, description="Description of the workflow step")
    step_status: Optional[str] = Field(None, description="Current status of the step")
    display_x: Optional[int] = Field(None, description="X coordinate for workflow editor")
    display_y: Optional[int] = Field(None, description="Y coordinate for workflow editor")
    commmmand_str: Optional[str] = Field(None, description="Command string for workflow step")
    step_type: WorkflowStepType = Field(..., description="Type of the workflow step")

class WorkflowStepCreate(WorkflowStepBase):
    """Schema for creating a new workflow step."""
    pass

class WorkflowStepResponse(WorkflowStepBase):
    """Schema for workflow step response."""
    step_id: int
    workflow_id: int

    class Config:
        from_attributes = True

class WorkflowBase(BaseModel):
    """Base schema for workflow data."""
    name: str = Field(..., description="Name of the workflow")

class WorkflowCreate(WorkflowBase):
    """Schema for creating a new workflow."""
    steps: Optional[List[WorkflowStepCreate]] = Field(default=[], description="List of steps in the workflow")

class WorkflowResponse(WorkflowBase):
    """Schema for workflow response."""
    workflow_id: int
    user_id: int
    steps: List[WorkflowStepResponse]

    class Config:
        from_attributes = True

class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""
    name: Optional[str] = Field(None, description="New name for the workflow")
    steps: Optional[List[WorkflowStepCreate]] = Field(None, description="Updated list of steps") 