"""Workflow execution engine and step processors."""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from app.models.database.workflow import (
    Workflow, WorkflowStep, WorkflowStepType, WorkflowRun, WorkflowStepResult,
    WorkflowRunStatus, create_workflow_run, update_workflow_run_status,
    create_workflow_step_result, update_workflow_step_result
)


class StepProcessor(ABC):
    """Abstract base class for workflow step processors."""
    
    @abstractmethod
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow step.
        
        Args:
            step: The workflow step to process
            input_data: Input data for the step
            
        Returns:
            Dict containing the output data from the step
            
        Raises:
            Exception: If step processing fails
        """
        pass


class TriggerProcessor(StepProcessor):
    """Processes TRIGGER steps that start workflows."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trigger step."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    command_config = {"instruction": step.command_str}
            
            return {
                "data": input_data,
                "trigger_config": command_config,
                "step_name": step.step_name
            }
        except Exception as e:
            raise Exception(f"Trigger step '{step.step_name}' failed: {str(e)}")


class FunctionalProcessor(StepProcessor):
    """Processes MAP, FILTER, REDUCE, and EXPAND steps using functional operations."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a functional step (map, filter, reduce, expand)."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    # If not JSON, treat as lambda expression
                    command_config = {"expression": step.command_str}
            
            data = input_data.get("data", input_data)
            
            try:
                step_type = WorkflowStepType(step.step_type)
            except ValueError:
                raise Exception(f"Unsupported functional step type: {step.step_type}")
            
            if step_type == WorkflowStepType.MAP:
                result = self._process_map(data, command_config)
            elif step_type == WorkflowStepType.FILTER:
                result = self._process_filter(data, command_config)
            elif step_type == WorkflowStepType.REDUCE:
                result = self._process_reduce(data, command_config)
            elif step_type == WorkflowStepType.EXPAND:
                result = self._process_expand(data, command_config)
            else:
                raise Exception(f"Unsupported functional step type: {step.step_type}")
            
            return {
                "data": result,
                "step_name": step.step_name,
                "operation_applied": command_config
            }
        except Exception as e:
            raise Exception(f"Functional step '{step.step_name}' failed: {str(e)}")
    
    def _process_map(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process map operation."""
        if not isinstance(data, list):
            return data
        
        # Simple field mapping
        if "field_mapping" in config:
            return [
                {config["field_mapping"].get(k, k): v for k, v in item.items()}
                if isinstance(item, dict) else item
                for item in data
            ]
        
        # Lambda-style expression (simplified)
        if "expression" in config:
            expr = config["expression"]
            # For safety, only allow simple field access
            if expr.startswith("x."):
                field = expr[2:]
                return [item.get(field) if isinstance(item, dict) else item for item in data]
        
        return data
    
    def _process_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process filter operation."""
        if not isinstance(data, list):
            return data
        
        if "condition" in config:
            condition = config["condition"]
            if "field" in condition and "value" in condition:
                return [
                    item for item in data
                    if isinstance(item, dict) and item.get(condition["field"]) == condition["value"]
                ]
        
        # Lambda-style expression (simplified)
        if "expression" in config:
            expr = config["expression"]
            # For safety, only allow simple comparisons
            if "==" in expr:
                field, value = expr.split("==")
                field = field.strip().replace("x.", "")
                value = value.strip().strip("'\"")
                return [
                    item for item in data
                    if isinstance(item, dict) and str(item.get(field)) == value
                ]
        
        return data
    
    def _process_reduce(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process reduce operation."""
        if not isinstance(data, list):
            return data
        
        if "operation" in config:
            op = config["operation"]
            if op == "count":
                return len(data)
            elif op == "sum" and "field" in config:
                field = config["field"]
                return sum(
                    item.get(field, 0) for item in data
                    if isinstance(item, dict) and isinstance(item.get(field), (int, float))
                )
            elif op == "max" and "field" in config:
                field = config["field"]
                values = [item.get(field) for item in data if isinstance(item, dict)]
                return max(values) if values else None
            elif op == "min" and "field" in config:
                field = config["field"]
                values = [item.get(field) for item in data if isinstance(item, dict)]
                return min(values) if values else None
        
        # Default to count
        return len(data)
    
    def _process_expand(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process expand operation."""
        if not isinstance(data, dict):
            return data
        
        if "field" in config:
            field = config["field"]
            if field in data and isinstance(data[field], list):
                return [
                    {**data, field: item} for item in data[field]
                ]
        
        return data


class CustomProcessor(StepProcessor):
    """Processes CUSTOM steps that execute custom code."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a custom step."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    command_config = {"code": step.command_str}
            
            data = input_data.get("data", input_data)
            
            # Simple example: if the command is a basic transformation
            if "message" in command_config:
                result = {
                    "original_data": data,
                    "custom_message": command_config["message"],
                    "processed_at": "custom_step"
                }
            else:
                result = data
            
            return {
                "data": result,
                "step_name": step.step_name,
                "custom_config": command_config
            }
        except Exception as e:
            raise Exception(f"Custom step '{step.step_name}' failed: {str(e)}")


class LLMProcessor(StepProcessor):
    """Processes LLM steps that use language models."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an LLM step."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    command_config = {"prompt": step.command_str}
            
            data = input_data.get("data", input_data)
            
            # For now, simulate LLM processing
            prompt = command_config.get("prompt", "Process this data")
            
            result = {
                "original_data": data,
                "llm_prompt": prompt,
                "llm_response": f"Processed data based on prompt: {prompt}",
                "mock_llm_output": True
            }
            
            return {
                "data": result,
                "step_name": step.step_name,
                "llm_config": command_config
            }
        except Exception as e:
            raise Exception(f"LLM step '{step.step_name}' failed: {str(e)}")


class AgentProcessor(StepProcessor):
    """Processes AGENT steps that run agent flows."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an agent step."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    command_config = {"agent": step.command_str}
            
            data = input_data.get("data", input_data)
            
            # Simulate agent processing
            agent_name = command_config.get("agent", "default_agent")
            
            result = {
                "original_data": data,
                "agent_name": agent_name,
                "agent_response": f"Agent {agent_name} processed the data",
                "mock_agent_output": True
            }
            
            return {
                "data": result,
                "step_name": step.step_name,
                "agent_config": command_config
            }
        except Exception as e:
            raise Exception(f"Agent step '{step.step_name}' failed: {str(e)}")


class AdapterProcessor(StepProcessor):
    """Processes ADAPTER steps that send data to external systems."""
    
    def process(self, step: WorkflowStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an adapter step."""
        try:
            command_config = {}
            if step.command_str:
                try:
                    command_config = json.loads(step.command_str)
                except json.JSONDecodeError:
                    command_config = {"adapter": step.command_str}
            
            data = input_data.get("data", input_data)
            
            # Simulate adapter processing
            adapter_name = command_config.get("adapter", "default_adapter")
            
            result = {
                "sent_data": data,
                "adapter_name": adapter_name,
                "adapter_response": f"Data sent to {adapter_name}",
                "mock_adapter_output": True
            }
            
            return {
                "data": result,
                "step_name": step.step_name,
                "adapter_config": command_config
            }
        except Exception as e:
            raise Exception(f"Adapter step '{step.step_name}' failed: {str(e)}")


class WorkflowExecutor:
    """Synchronous workflow execution engine."""
    
    def __init__(self):
        functional_processor = FunctionalProcessor()
        self.processors = {
            WorkflowStepType.TRIGGER: TriggerProcessor(),
            WorkflowStepType.MAP: functional_processor,
            WorkflowStepType.FILTER: functional_processor,
            WorkflowStepType.REDUCE: functional_processor,
            WorkflowStepType.EXPAND: functional_processor,
            WorkflowStepType.CUSTOM: CustomProcessor(),
            WorkflowStepType.LLM: LLMProcessor(),
            WorkflowStepType.AGENT: AgentProcessor(),
            WorkflowStepType.ADAPTER: AdapterProcessor(),
        }
    
    def execute_workflow(self, workflow: Workflow, user_id: int, input_data: Dict[str, Any] = None) -> WorkflowRun:
        """Execute a workflow synchronously."""
        # Create workflow run record
        run = create_workflow_run(workflow.workflow_id, user_id, input_data)
        
        try:
            # Sort steps by display position (simple ordering for now)
            sorted_steps = sorted(workflow.steps, key=lambda s: (s.display_y, s.display_x))
            
            # Execute each step in sequence
            current_data = input_data or {}
            step_results = []
            
            for step in sorted_steps:
                try:
                    # Create step result record
                    step_result = create_workflow_step_result(run.run_id, step.step_id, current_data)
                    
                    # Get the appropriate processor
                    step_type = WorkflowStepType(step.step_type)
                    processor = self.processors.get(step_type)
                    
                    if not processor:
                        raise Exception(f"No processor found for step type: {step.step_type}")
                    
                    # Process the step
                    output_data = processor.process(step, current_data)
                    
                    # Update step result
                    update_workflow_step_result(
                        step_result.result_id, 
                        WorkflowRunStatus.COMPLETED, 
                        output_data
                    )
                    
                    # Use this step's output as input for next step
                    current_data = output_data
                    step_results.append(step_result)
                    
                except Exception as e:
                    # Update step result with error
                    update_workflow_step_result(
                        step_result.result_id, 
                        WorkflowRunStatus.FAILED, 
                        error_message=str(e)
                    )
                    raise e
            
            # Update workflow run as completed
            update_workflow_run_status(
                run.run_id, 
                WorkflowRunStatus.COMPLETED, 
                output_data=current_data
            )
            
            return run
            
        except Exception as e:
            # Update workflow run as failed
            update_workflow_run_status(
                run.run_id, 
                WorkflowRunStatus.FAILED, 
                error_message=str(e)
            )
            raise e