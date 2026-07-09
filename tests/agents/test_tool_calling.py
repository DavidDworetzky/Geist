"""Tests for robust tool-call parsing, validation, and dispatch."""
import json

import pytest

from adapters.adapter_registry import AdapterWrapper
from adapters.base_adapter import BaseAdapter
from adapters.tool_schema import enumerate_tool_schemas
from agents.tool_calling import (
    ToolCall,
    ToolCallError,
    ToolDispatcher,
    extract_json_candidates,
    parse_tool_call,
    run_prompt_tool_call,
    validate_tool_call,
)


class EchoAdapter(BaseAdapter):
    """Test adapter with typed actions and a failing action."""

    def __init__(self):
        self.calls = []

    def enumerate_actions(self):
        return ["echo", "add", "explode"]

    def echo(self, message: str) -> str:
        """Echo a message back."""
        self.calls.append(("echo", message))
        return f"echo: {message}"

    def add(self, a: int, b: int) -> int:
        """Add two integers."""
        self.calls.append(("add", a, b))
        return a + b

    def explode(self) -> None:
        """Always raises."""
        raise RuntimeError("boom")


@pytest.fixture()
def adapter():
    return EchoAdapter()


@pytest.fixture()
def schemas(adapter):
    return enumerate_tool_schemas(adapter)


@pytest.fixture()
def dispatcher(adapter, schemas):
    return ToolDispatcher(
        [AdapterWrapper(name="EchoAdapter", instance=adapter)],
        schemas=schemas,
        function_log=[],
    )


class TestParseToolCall:
    def test_parses_plain_json(self):
        call = parse_tool_call('{"class": "EchoAdapter", "function": "echo", "parameters": {"message": "hi"}}')
        assert call.adapter == "EchoAdapter"
        assert call.function == "echo"
        assert call.arguments == {"message": "hi"}

    def test_parses_fenced_json(self):
        text = 'Sure, here is the call:\n```json\n{"class": "EchoAdapter", "function": "echo", "parameters": {"message": "hi"}}\n```'
        call = parse_tool_call(text)
        assert call.function == "echo"

    def test_parses_json_embedded_in_prose(self):
        text = 'I will call the tool now. {"class": "EchoAdapter", "function": "add", "parameters": {"a": 1, "b": 2}} That should work.'
        call = parse_tool_call(text)
        assert call.function == "add"
        assert call.arguments == {"a": 1, "b": 2}

    def test_accepts_alias_keys(self):
        call = parse_tool_call('{"adapter": "EchoAdapter", "action": "echo", "arguments": {"message": "hi"}}')
        assert call.adapter == "EchoAdapter"
        assert call.function == "echo"

    def test_accepts_qualified_name(self):
        call = parse_tool_call('{"name": "EchoAdapter__echo", "arguments": {"message": "hi"}}')
        assert call.adapter == "EchoAdapter"
        assert call.function == "echo"

    def test_missing_parameters_defaults_to_empty(self):
        call = parse_tool_call('{"class": "EchoAdapter", "function": "explode"}')
        assert call.arguments == {}

    def test_rejects_empty_text(self):
        with pytest.raises(ToolCallError):
            parse_tool_call("")

    def test_rejects_text_without_json(self):
        with pytest.raises(ToolCallError):
            parse_tool_call("I could not decide which tool to use.")

    def test_rejects_non_object_parameters(self):
        with pytest.raises(ToolCallError):
            parse_tool_call('{"class": "EchoAdapter", "function": "echo", "parameters": [1, 2]}')

    def test_skips_invalid_json_and_finds_valid_object(self):
        text = '{"broken": } then {"class": "EchoAdapter", "function": "echo", "parameters": {"message": "hi"}}'
        call = parse_tool_call(text)
        assert call.function == "echo"

    def test_extract_candidates_handles_braces_in_strings(self):
        candidates = extract_json_candidates('{"key": "value with } brace"}')
        assert candidates == ['{"key": "value with } brace"}']


class TestValidateToolCall:
    def test_unknown_tool_lists_available(self, schemas):
        call = ToolCall(adapter="EchoAdapter", function="nope", arguments={})
        with pytest.raises(ToolCallError) as excinfo:
            validate_tool_call(call, schemas)
        assert "EchoAdapter__echo" in str(excinfo.value)

    def test_missing_required_argument(self, schemas):
        call = ToolCall(adapter="EchoAdapter", function="echo", arguments={})
        with pytest.raises(ToolCallError) as excinfo:
            validate_tool_call(call, schemas)
        assert "message" in str(excinfo.value)

    def test_coerces_string_numbers(self, schemas):
        call = ToolCall(adapter="EchoAdapter", function="add", arguments={"a": "3", "b": 4})
        normalized = validate_tool_call(call, schemas)
        assert normalized.arguments == {"a": 3, "b": 4}

    def test_case_insensitive_match_normalizes(self, schemas):
        call = ToolCall(adapter="echoadapter", function="ECHO", arguments={"message": "hi"})
        normalized = validate_tool_call(call, schemas)
        assert normalized.adapter == "EchoAdapter"
        assert normalized.function == "echo"

    def test_drops_unknown_arguments(self, schemas):
        call = ToolCall(
            adapter="EchoAdapter",
            function="echo",
            arguments={"message": "hi", "hallucinated": True},
        )
        normalized = validate_tool_call(call, schemas)
        assert normalized.arguments == {"message": "hi"}

    def test_uncoercible_type_errors(self, schemas):
        call = ToolCall(adapter="EchoAdapter", function="add", arguments={"a": "not-a-number", "b": 1})
        with pytest.raises(ToolCallError) as excinfo:
            validate_tool_call(call, schemas)
        assert "'a'" in str(excinfo.value)


class TestToolDispatcher:
    def test_dispatch_success(self, dispatcher, adapter):
        result = dispatcher.dispatch(
            ToolCall(adapter="EchoAdapter", function="echo", arguments={"message": "hi"})
        )
        assert result.success
        assert result.result == "echo: hi"
        assert adapter.calls == [("echo", "hi")]

    def test_dispatch_captures_adapter_exceptions(self, dispatcher):
        result = dispatcher.dispatch(ToolCall(adapter="EchoAdapter", function="explode", arguments={}))
        assert not result.success
        assert "RuntimeError" in result.error

    def test_dispatch_text_captures_parse_errors(self, dispatcher):
        result = dispatcher.dispatch_text("no json here")
        assert not result.success
        assert result.call is None

    def test_dispatch_journals_to_function_log(self, dispatcher):
        dispatcher.dispatch(ToolCall(adapter="EchoAdapter", function="echo", arguments={"message": "hi"}))
        assert len(dispatcher.function_log) == 1
        entry = json.loads(dispatcher.function_log[0])
        assert entry["adapter"] == "EchoAdapter"
        assert entry["function"] == "echo"
        assert entry["success"] is True

    def test_result_to_content_is_json(self, dispatcher):
        result = dispatcher.dispatch(
            ToolCall(adapter="EchoAdapter", function="add", arguments={"a": 1, "b": 2})
        )
        assert json.loads(result.to_content()) == {"success": True, "result": 3}


class TestRunPromptToolCall:
    def test_retries_with_validation_feedback(self, schemas, dispatcher, adapter):
        responses = iter([
            "I think I should use the echo tool.",
            '{"class": "EchoAdapter", "function": "echo", "parameters": {"message": "hello"}}',
        ])
        prompts_seen = []

        def complete_fn(prompt, system_prompt):
            prompts_seen.append(prompt)
            return next(responses)

        result = run_prompt_tool_call(complete_fn, schemas, dispatcher, "Say hello")

        assert result.success
        assert result.result == "echo: hello"
        assert len(prompts_seen) == 2
        assert "previous response was invalid" in prompts_seen[1]
        # tool schemas are visible in every attempt prompt
        assert "AVAILABLE_TOOLS" in prompts_seen[0]
        assert "EchoAdapter" in prompts_seen[0]

    def test_gives_up_after_max_attempts(self, schemas, dispatcher):
        result = run_prompt_tool_call(
            lambda prompt, system: "never valid",
            schemas,
            dispatcher,
            "Say hello",
            max_attempts=2,
        )
        assert not result.success
        assert "after 2 attempts" in result.error

    def test_no_tools_available(self, dispatcher):
        result = run_prompt_tool_call(lambda p, s: "", [], dispatcher, "Say hello")
        assert not result.success
        assert "No tools" in result.error
