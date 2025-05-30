import pytest
from app.models.database.workflow import WorkflowStepType
from app.schemas.workflow import WorkflowStepCreate, WorkflowCreate
from pydantic import ValidationError


def test_workflow_step_types():
    """Test that all workflow step types are valid."""
    # Test all step types including new trigger and adapter
    valid_types = [
        WorkflowStepType.TRIGGER,
        WorkflowStepType.MAP,
        WorkflowStepType.FILTER,
        WorkflowStepType.REDUCE,
        WorkflowStepType.EXPAND,
        WorkflowStepType.CUSTOM,
        WorkflowStepType.LLM,
        WorkflowStepType.AGENT,
        WorkflowStepType.ADAPTER,
    ]
    
    for step_type in valid_types:
        assert step_type.value in [
            "trigger", "map", "filter", "reduce", 
            "expand", "custom", "llm", "agent", "adapter"
        ]


def test_create_trigger_step():
    """Test creating a trigger (input) workflow step."""
    trigger_step = WorkflowStepCreate(
        step_name="HTTP Webhook Trigger",
        step_description="Receives HTTP POST requests to start workflow",
        step_type=WorkflowStepType.TRIGGER,
        command_str="webhook:post:/api/webhook/start"
    )
    
    assert trigger_step.step_name == "HTTP Webhook Trigger"
    assert trigger_step.step_type == WorkflowStepType.TRIGGER


def test_create_adapter_step():
    """Test creating an adapter (output) workflow step."""
    adapter_step = WorkflowStepCreate(
        step_name="Email Adapter",
        step_description="Sends results via email",
        step_type=WorkflowStepType.ADAPTER,
        command_str="email:send:recipient@example.com"
    )
    
    assert adapter_step.step_name == "Email Adapter"
    assert adapter_step.step_type == WorkflowStepType.ADAPTER


def test_workflow_with_trigger_and_adapter():
    """Test creating a complete workflow with trigger input and adapter output."""
    workflow = WorkflowCreate(
        name="Data Processing Pipeline",
        steps=[
            WorkflowStepCreate(
                step_name="API Trigger",
                step_description="Receives data from external API",
                step_type=WorkflowStepType.TRIGGER,
                display_x=100,
                display_y=100
            ),
            WorkflowStepCreate(
                step_name="Process Data",
                step_description="Transform and validate data",
                step_type=WorkflowStepType.CUSTOM,
                display_x=300,
                display_y=100
            ),
            WorkflowStepCreate(
                step_name="Database Adapter",
                step_description="Store results in database",
                step_type=WorkflowStepType.ADAPTER,
                display_x=500,
                display_y=100
            )
        ]
    )
    
    assert workflow.name == "Data Processing Pipeline"
    assert len(workflow.steps) == 3
    assert workflow.steps[0].step_type == WorkflowStepType.TRIGGER
    assert workflow.steps[2].step_type == WorkflowStepType.ADAPTER 