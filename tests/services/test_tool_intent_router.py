from __future__ import annotations

import pytest

from agents.models.tool_calling import ChatMessage, ModelEvent, ModelTurn
from app.services.tool_intent_router import (
    ToolIntentRouter,
    parse_tool_intent,
    parse_tool_intent_decision,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"intent":"answer"}', "answer"),
        ('{"intent":"Sensitive_Answer"}', "sensitive_answer"),
        ("The intent is action.", "action"),
        ('```json\n{"intent":"image_generation"}\n```', "image_generation"),
        ('{"intent":"unknown"}', "answer"),
        ("It could be action or answer.", "answer"),
    ],
)
def test_parse_tool_intent(response, expected):
    assert parse_tool_intent(response) == expected


@pytest.mark.parametrize(
    ("response", "expected_intent", "expected_retrieval"),
    [
        ('{"intent":"answer","needs_retrieval":true}', "answer", True),
        ('{"intent":"answer","needs_retrieval":false}', "answer", False),
        ('{"intent":"sensitive_answer"}', "sensitive_answer", False),
        ('{"intent":"answer","needs_retrieval":"true"}', "answer", False),
        ('```json\n{"intent":"answer","needs_retrieval":true}\n```', "answer", True),
    ],
)
def test_parse_tool_intent_decision(response, expected_intent, expected_retrieval):
    decision = parse_tool_intent_decision(response)

    assert decision.intent == expected_intent
    assert decision.needs_retrieval is expected_retrieval


class ClassifierBackend:
    def __init__(self, response: str):
        self.response = response
        self.requests = []

    def stream_model_turn(self, messages, tools, config):
        self.requests.append((messages, tools, config))
        yield ModelEvent.turn_complete(ModelTurn(text=self.response, finish_reason="stop"))


def test_router_runs_tool_free_turn_with_recent_conversation():
    backend = ClassifierBackend('{"intent":"sensitive_answer","needs_retrieval":true}')
    messages = [
        ChatMessage(role="system", content="Assistant prompt"),
        ChatMessage(role="user", content="Earlier question"),
        ChatMessage(role="assistant", content="Earlier answer"),
        ChatMessage(role="tool", content="hidden tool result"),
        ChatMessage(role="user", content="Find my medical notes"),
    ]

    decision = ToolIntentRouter().classify(backend, messages)

    assert decision.intent == "sensitive_answer"
    assert decision.needs_retrieval is True
    classifier_messages, tools, config = backend.requests[0]
    assert tools == []
    assert classifier_messages[0].role == "system"
    assert [message.role for message in classifier_messages[1:]] == [
        "user",
        "assistant",
        "user",
    ]
    assert classifier_messages[-1].content == "Find my medical notes"
    assert config.temperature == 0.0
    assert config.max_tokens == 128
