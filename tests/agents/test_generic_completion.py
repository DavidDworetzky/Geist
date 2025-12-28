"""
Unit tests for GenericCompletion data types.

Tests verify that from_dict methods handle arbitrary extra fields gracefully,
ensuring forward compatibility with API changes.
"""
import pytest
from agents.models.generic_completion import (
    Message,
    Choice,
    TokenDetails,
    Usage,
    GenericCompletion
)


class TestMessageFromDict:
    """Test Message.from_dict handles extra fields."""

    def test_basic_message(self):
        """Test basic message parsing."""
        data = {
            "role": "assistant",
            "content": "Hello, world!",
            "refusal": None
        }
        message = Message.from_dict(data)
        assert message.role == "assistant"
        assert message.content == "Hello, world!"
        assert message.refusal is None

    def test_message_with_extra_fields(self):
        """Test message ignores unknown fields like 'annotations'."""
        data = {
            "role": "assistant",
            "content": "Test response",
            "refusal": None,
            "annotations": [{"type": "url", "start": 0, "end": 10}],
            "audio": {"id": "audio_123"},
            "future_field": "some_value"
        }
        message = Message.from_dict(data)
        assert message.role == "assistant"
        assert message.content == "Test response"
        assert message.refusal is None
        assert not hasattr(message, "annotations")
        assert not hasattr(message, "audio")
        assert not hasattr(message, "future_field")

    def test_message_minimal_fields(self):
        """Test message with only required fields."""
        data = {
            "role": "user",
            "content": "Hello"
        }
        message = Message.from_dict(data)
        assert message.role == "user"
        assert message.content == "Hello"


class TestChoiceFromDict:
    """Test Choice.from_dict handles extra fields."""

    def test_basic_choice(self):
        """Test basic choice parsing."""
        data = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response"
            },
            "logprobs": None,
            "finish_reason": "stop"
        }
        choice = Choice.from_dict(data)
        assert choice.index == 0
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"

    def test_choice_with_extra_fields(self):
        """Test choice ignores unknown fields."""
        data = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response",
                "annotations": []
            },
            "logprobs": None,
            "finish_reason": "stop",
            "content_filter_results": {"hate": {"filtered": False}},
            "future_field": 123
        }
        choice = Choice.from_dict(data)
        assert choice.index == 0
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"
        assert not hasattr(choice, "content_filter_results")
        assert not hasattr(choice, "future_field")

    def test_choice_already_parsed(self):
        """Test choice returns itself if already a Choice object."""
        original = Choice(
            index=0,
            message=Message(role="assistant", content="Test"),
            finish_reason="stop"
        )
        result = Choice.from_dict(original)
        assert result is original


class TestTokenDetailsFromDict:
    """Test TokenDetails.from_dict handles extra fields."""

    def test_basic_token_details(self):
        """Test basic token details parsing."""
        data = {
            "cached_tokens": 100,
            "reasoning_tokens": 50,
            "audio_tokens": 0
        }
        details = TokenDetails.from_dict(data)
        assert details.cached_tokens == 100
        assert details.reasoning_tokens == 50
        assert details.audio_tokens == 0

    def test_token_details_with_extra_fields(self):
        """Test token details ignores unknown fields."""
        data = {
            "cached_tokens": 100,
            "reasoning_tokens": 50,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 25,
            "rejected_prediction_tokens": 5,
            "future_token_type": 999
        }
        details = TokenDetails.from_dict(data)
        assert details.cached_tokens == 100
        assert details.reasoning_tokens == 50
        assert details.audio_tokens == 0
        assert not hasattr(details, "accepted_prediction_tokens")
        assert not hasattr(details, "rejected_prediction_tokens")
        assert not hasattr(details, "future_token_type")

    def test_token_details_partial_fields(self):
        """Test token details with partial fields uses defaults."""
        data = {
            "cached_tokens": 50
        }
        details = TokenDetails.from_dict(data)
        assert details.cached_tokens == 50
        assert details.reasoning_tokens == 0
        assert details.audio_tokens == 0


class TestUsageFromDict:
    """Test Usage.from_dict handles extra fields."""

    def test_basic_usage(self):
        """Test basic usage parsing."""
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0}
        }
        usage = Usage.from_dict(data)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_with_extra_fields(self):
        """Test usage ignores unknown fields."""
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 5,
            "future_usage_field": "value"
        }
        usage = Usage.from_dict(data)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert not hasattr(usage, "prompt_cache_hit_tokens")
        assert not hasattr(usage, "prompt_cache_miss_tokens")
        assert not hasattr(usage, "future_usage_field")

    def test_usage_already_parsed(self):
        """Test usage returns itself if already a Usage object."""
        original = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_tokens_details=TokenDetails(),
            completion_tokens_details=TokenDetails()
        )
        result = Usage.from_dict(original)
        assert result is original


class TestGenericCompletionFromDict:
    """Test GenericCompletion.from_dict handles extra fields."""

    def test_basic_completion(self):
        """Test basic completion parsing."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1728753308,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }
        completion = GenericCompletion.from_dict(data)
        assert completion.id == "chatcmpl-123"
        assert completion.model == "gpt-4o"
        assert len(completion.choices) == 1
        assert completion.choices[0].message.content == "Hello"

    def test_completion_with_extra_fields(self):
        """Test completion ignores unknown fields like 'service_tier'."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1728753308,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "annotations": []
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            },
            "service_tier": "default",
            "x_groq": {"id": "req_123"},
            "future_top_level_field": True
        }
        completion = GenericCompletion.from_dict(data)
        assert completion.id == "chatcmpl-123"
        assert completion.model == "gpt-4o"
        assert completion.choices[0].message.content == "Hello"
        assert not hasattr(completion, "service_tier")
        assert not hasattr(completion, "x_groq")
        assert not hasattr(completion, "future_top_level_field")

    def test_completion_with_system_fingerprint(self):
        """Test completion with optional system_fingerprint."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1728753308,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            },
            "system_fingerprint": "fp_abc123"
        }
        completion = GenericCompletion.from_dict(data)
        assert completion.system_fingerprint == "fp_abc123"


class TestRealWorldAPIResponses:
    """Test with real-world API response structures."""

    def test_openai_gpt4o_response_with_annotations(self):
        """Test OpenAI GPT-4o response that includes annotations field."""
        data = {
            "id": "chatcmpl-AfQnpFJyAzGLHLZB1ENmXRF0yYWaD",
            "object": "chat.completion",
            "created": 1735335789,
            "model": "gpt-4o-2024-11-20",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                    "refusal": None,
                    "annotations": []
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "total_tokens": 29,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "service_tier": "default",
            "system_fingerprint": "fp_d28bcae782"
        }
        completion = GenericCompletion.from_dict(data)
        assert completion.id == "chatcmpl-AfQnpFJyAzGLHLZB1ENmXRF0yYWaD"
        assert completion.model == "gpt-4o-2024-11-20"
        assert completion.choices[0].message.content == "Hello! How can I help you today?"
        assert completion.system_fingerprint == "fp_d28bcae782"
        assert not hasattr(completion, "service_tier")

    def test_groq_response_with_x_groq(self):
        """Test Groq API response with x_groq field."""
        data = {
            "id": "chatcmpl-groq-123",
            "object": "chat.completion",
            "created": 1735335789,
            "model": "llama-3.1-70b-versatile",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response from Groq"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            },
            "x_groq": {
                "id": "req_01jfw9t6t2ey9r9e9v97hgp96m",
                "usage": {
                    "queue_time": 0.012,
                    "prompt_time": 0.05,
                    "completion_time": 0.1
                }
            }
        }
        completion = GenericCompletion.from_dict(data)
        assert completion.id == "chatcmpl-groq-123"
        assert completion.choices[0].message.content == "Response from Groq"
        assert not hasattr(completion, "x_groq")
