import pytest
from unittest.mock import Mock, patch
from app.services.workflow_execution import (
    FunctionalProcessor, WorkflowExecutor, WorkflowStepType
)
from app.models.database.workflow import Workflow, WorkflowStep


class TestFunctionalProcessor:
    """Test class for functional operations in workflow execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FunctionalProcessor()
    
    def test_map_with_field_mapping(self):
        """Test map operation with field mapping configuration."""
        # Create a mock workflow step
        step = Mock()
        step.step_name = "test_map"
        step.step_type = WorkflowStepType.MAP.value
        step.command_str = '{"field_mapping": {"old_field": "new_field", "value": "amount"}}'
        
        # Test data
        input_data = {
            "data": [
                {"old_field": "test1", "value": 100},
                {"old_field": "test2", "value": 200}
            ]
        }
        
        # Execute the step
        result = self.processor.process(step, input_data)
        
        # Verify the result
        assert result["step_name"] == "test_map"
        assert "data" in result
        assert len(result["data"]) == 2
        assert result["data"][0] == {"new_field": "test1", "amount": 100}
        assert result["data"][1] == {"new_field": "test2", "amount": 200}
    
    def test_map_with_expression(self):
        """Test map operation with lambda-style expression."""
        step = Mock()
        step.step_name = "test_map_expr"
        step.step_type = WorkflowStepType.MAP.value
        step.command_str = '{"expression": "x.name"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == ["Alice", "Bob"]
    
    def test_map_with_non_list_data(self):
        """Test map operation with non-list data returns data unchanged."""
        step = Mock()
        step.step_name = "test_map_nonlist"
        step.step_type = WorkflowStepType.MAP.value
        step.command_str = '{"expression": "x.name"}'
        
        input_data = {"data": "not a list"}
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == "not a list"
    
    def test_filter_with_condition(self):
        """Test filter operation with field condition."""
        step = Mock()
        step.step_name = "test_filter"
        step.step_type = WorkflowStepType.FILTER.value
        step.command_str = '{"condition": {"field": "status", "value": "active"}}'
        
        input_data = {
            "data": [
                {"name": "Alice", "status": "active"},
                {"name": "Bob", "status": "inactive"},
                {"name": "Charlie", "status": "active"}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert len(result["data"]) == 2
        assert result["data"][0]["name"] == "Alice"
        assert result["data"][1]["name"] == "Charlie"
    
    def test_filter_with_expression(self):
        """Test filter operation with lambda-style expression."""
        step = Mock()
        step.step_name = "test_filter_expr"
        step.step_type = WorkflowStepType.FILTER.value
        step.command_str = '{"expression": "x.age == 30"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 30}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert len(result["data"]) == 2
        assert result["data"][0]["name"] == "Alice"
        assert result["data"][1]["name"] == "Charlie"
    
    def test_filter_with_non_list_data(self):
        """Test filter operation with non-list data returns data unchanged."""
        step = Mock()
        step.step_name = "test_filter_nonlist"
        step.step_type = WorkflowStepType.FILTER.value
        step.command_str = '{"condition": {"field": "status", "value": "active"}}'
        
        input_data = {"data": {"name": "Alice", "status": "active"}}
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == {"name": "Alice", "status": "active"}
    
    def test_reduce_count_operation(self):
        """Test reduce operation with count."""
        step = Mock()
        step.step_name = "test_reduce_count"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{"operation": "count"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "value": 10},
                {"name": "Bob", "value": 20},
                {"name": "Charlie", "value": 30}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == 3
    
    def test_reduce_sum_operation(self):
        """Test reduce operation with sum."""
        step = Mock()
        step.step_name = "test_reduce_sum"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{"operation": "sum", "field": "value"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "value": 10},
                {"name": "Bob", "value": 20},
                {"name": "Charlie", "value": 30}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == 60
    
    def test_reduce_max_operation(self):
        """Test reduce operation with max."""
        step = Mock()
        step.step_name = "test_reduce_max"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{"operation": "max", "field": "value"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "value": 10},
                {"name": "Bob", "value": 30},
                {"name": "Charlie", "value": 20}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == 30
    
    def test_reduce_min_operation(self):
        """Test reduce operation with min."""
        step = Mock()
        step.step_name = "test_reduce_min"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{"operation": "min", "field": "value"}'
        
        input_data = {
            "data": [
                {"name": "Alice", "value": 10},
                {"name": "Bob", "value": 30},
                {"name": "Charlie", "value": 20}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == 10
    
    def test_reduce_default_to_count(self):
        """Test reduce operation defaults to count when no operation specified."""
        step = Mock()
        step.step_name = "test_reduce_default"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{}'
        
        input_data = {
            "data": [
                {"name": "Alice"},
                {"name": "Bob"}
            ]
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == 2
    
    def test_reduce_with_non_list_data(self):
        """Test reduce operation with non-list data returns data unchanged."""
        step = Mock()
        step.step_name = "test_reduce_nonlist"
        step.step_type = WorkflowStepType.REDUCE.value
        step.command_str = '{"operation": "count"}'
        
        input_data = {"data": {"name": "Alice"}}
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == {"name": "Alice"}
    
    def test_expand_operation(self):
        """Test expand operation with field expansion."""
        step = Mock()
        step.step_name = "test_expand"
        step.step_type = WorkflowStepType.EXPAND.value
        step.command_str = '{"field": "items"}'
        
        input_data = {
            "data": {
                "order_id": "12345",
                "customer": "Alice",
                "items": ["apple", "banana", "orange"]
            }
        }
        
        result = self.processor.process(step, input_data)
        
        assert len(result["data"]) == 3
        assert result["data"][0] == {"order_id": "12345", "customer": "Alice", "items": "apple"}
        assert result["data"][1] == {"order_id": "12345", "customer": "Alice", "items": "banana"}
        assert result["data"][2] == {"order_id": "12345", "customer": "Alice", "items": "orange"}
    
    def test_expand_with_non_dict_data(self):
        """Test expand operation with non-dict data returns data unchanged."""
        step = Mock()
        step.step_name = "test_expand_nondict"
        step.step_type = WorkflowStepType.EXPAND.value
        step.command_str = '{"field": "items"}'
        
        input_data = {"data": ["apple", "banana"]}
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == ["apple", "banana"]
    
    def test_expand_with_missing_field(self):
        """Test expand operation with missing field returns data unchanged."""
        step = Mock()
        step.step_name = "test_expand_missing"
        step.step_type = WorkflowStepType.EXPAND.value
        step.command_str = '{"field": "missing_field"}'
        
        input_data = {
            "data": {
                "order_id": "12345",
                "customer": "Alice"
            }
        }
        
        result = self.processor.process(step, input_data)
        
        assert result["data"] == {"order_id": "12345", "customer": "Alice"}
    
    def test_error_handling_invalid_json(self):
        """Test error handling for invalid JSON in command_str."""
        step = Mock()
        step.step_name = "test_error"
        step.step_type = WorkflowStepType.MAP.value
        step.command_str = 'invalid json'
        
        input_data = {"data": [{"name": "Alice"}]}
        
        result = self.processor.process(step, input_data)
        
        # Should treat as expression
        assert "operation_applied" in result
        assert result["operation_applied"]["expression"] == "invalid json"
    
    def test_error_handling_unsupported_step_type(self):
        """Test error handling for unsupported step type."""
        step = Mock()
        step.step_name = "test_unsupported"
        step.step_type = "UNSUPPORTED"
        step.command_str = '{}'
        
        input_data = {"data": []}
        
        with pytest.raises(Exception) as exc_info:
            self.processor.process(step, input_data)
        
        assert "Unsupported functional step type" in str(exc_info.value)


class TestWorkflowExecutor:
    """Test class for workflow executor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = WorkflowExecutor()
    
    @patch('app.services.workflow_execution.create_workflow_run')
    @patch('app.services.workflow_execution.create_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_run_status')
    def test_execute_workflow_with_map_step(self, mock_update_run, mock_update_step, 
                                           mock_create_step, mock_create_run):
        """Test workflow execution with a map step."""
        # Mock the workflow run
        mock_run = Mock()
        mock_run.run_id = 1
        mock_create_run.return_value = mock_run
        
        # Mock the step result
        mock_step_result = Mock()
        mock_step_result.result_id = 1
        mock_create_step.return_value = mock_step_result
        
        # Create a mock workflow with a map step
        workflow = Mock()
        workflow.workflow_id = 1
        workflow.steps = [
            Mock(
                step_id=1,
                step_name="map_names",
                step_type=WorkflowStepType.MAP.value,
                command_str='{"expression": "x.name"}',
                display_x=0,
                display_y=0
            )
        ]
        
        input_data = {
            "data": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        }
        
        # Execute the workflow
        result = self.executor.execute_workflow(workflow, user_id=1, input_data=input_data)
        
        # Verify the mocks were called correctly
        mock_create_run.assert_called_once_with(1, 1, input_data)
        mock_create_step.assert_called_once_with(1, 1, input_data)
        mock_update_step.assert_called_once()
        mock_update_run.assert_called_once()
        
        assert result == mock_run
    
    @patch('app.services.workflow_execution.create_workflow_run')
    @patch('app.services.workflow_execution.create_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_run_status')
    def test_execute_workflow_with_multiple_steps(self, mock_update_run, mock_update_step, 
                                                 mock_create_step, mock_create_run):
        """Test workflow execution with multiple functional steps."""
        # Mock the workflow run
        mock_run = Mock()
        mock_run.run_id = 1
        mock_create_run.return_value = mock_run
        
        # Mock the step result
        mock_step_result = Mock()
        mock_step_result.result_id = 1
        mock_create_step.return_value = mock_step_result
        
        # Create a mock workflow with filter -> reduce steps
        workflow = Mock()
        workflow.workflow_id = 1
        workflow.steps = [
            Mock(
                step_id=1,
                step_name="filter_active",
                step_type=WorkflowStepType.FILTER.value,
                command_str='{"condition": {"field": "status", "value": "active"}}',
                display_x=0,
                display_y=0
            ),
            Mock(
                step_id=2,
                step_name="count_results",
                step_type=WorkflowStepType.REDUCE.value,
                command_str='{"operation": "count"}',
                display_x=0,
                display_y=1
            )
        ]
        
        input_data = {
            "data": [
                {"name": "Alice", "status": "active"},
                {"name": "Bob", "status": "inactive"},
                {"name": "Charlie", "status": "active"}
            ]
        }
        
        # Execute the workflow
        result = self.executor.execute_workflow(workflow, user_id=1, input_data=input_data)
        
        # Verify the mocks were called correctly
        mock_create_run.assert_called_once_with(1, 1, input_data)
        assert mock_create_step.call_count == 2
        assert mock_update_step.call_count == 2
        mock_update_run.assert_called_once()
        
        assert result == mock_run
    
    @patch('app.services.workflow_execution.create_workflow_run')
    @patch('app.services.workflow_execution.create_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_step_result')
    @patch('app.services.workflow_execution.update_workflow_run_status')
    def test_execute_workflow_step_failure(self, mock_update_run, mock_update_step, 
                                          mock_create_step, mock_create_run):
        """Test workflow execution with step failure."""
        # Mock the workflow run
        mock_run = Mock()
        mock_run.run_id = 1
        mock_create_run.return_value = mock_run
        
        # Mock the step result
        mock_step_result = Mock()
        mock_step_result.result_id = 1
        mock_create_step.return_value = mock_step_result
        
        # Create a mock workflow with an unsupported step type
        workflow = Mock()
        workflow.workflow_id = 1
        workflow.steps = [
            Mock(
                step_id=1,
                step_name="unsupported_step",
                step_type="UNSUPPORTED",
                command_str='{}',
                display_x=0,
                display_y=0
            )
        ]
        
        input_data = {"data": []}
        
        # Execute the workflow - should raise exception
        with pytest.raises(Exception):
            self.executor.execute_workflow(workflow, user_id=1, input_data=input_data)
        
        # Verify error handling was called
        mock_create_run.assert_called_once_with(1, 1, input_data)
        mock_create_step.assert_called_once_with(1, 1, input_data)
        mock_update_run.assert_called_once()  # Should be called to mark as failed