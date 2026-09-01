"""Small model-backed classifier for per-turn tool catalog selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ToolIntent,
)


TOOL_INTENT_PROMPT = """Classify the final user request into exactly one intent and decide whether answering it requires retrieval.

- answer: direct answers, writing, public information, current news, web search, and ordinary information retrieval.
- sensitive_answer: retrieving documented or uploaded information, past memories, or anything involving a legal, personal, or medical topic. This route is local-only.
- action: code generation, computer use, sending or changing something, writing files, or performing another task.
- image_generation: explicitly generating, drawing, creating, or editing an image.

Choose the requested outcome, not words mentioned incidentally. Writing a poem or haiku about an image is answer, not image_generation. Sending medical information is action, while finding or discussing it is sensitive_answer.

Set needs_retrieval to true only when the answer depends on current public information or user-owned information that is not already in the conversation. Writing, rewriting, general explanations, and reasoning use false.

Return only JSON in this exact shape: {"intent":"answer","needs_retrieval":false}
"""

_INTENTS: frozenset[str] = frozenset({"answer", "sensitive_answer", "action", "image_generation"})
_INTENT_PATTERN = re.compile(
    r"(?<![a-z_])(sensitive_answer|image_generation|action|answer)(?![a-z_])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolIntentDecision:
    intent: ToolIntent = "action"
    needs_retrieval: bool = False


def parse_tool_intent_decision(response: str) -> ToolIntentDecision:
    """Parse a classifier response, defaulting ambiguous output to Action."""

    normalized = response.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        fenced_lines = normalized.splitlines()
        if len(fenced_lines) >= 3 and fenced_lines[-1].strip() == "```":
            normalized = "\n".join(fenced_lines[1:-1]).strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        matches = {match.lower() for match in _INTENT_PATTERN.findall(normalized)}
        if len(matches) == 1:
            return ToolIntentDecision(intent=next(iter(matches)))
        return ToolIntentDecision()

    if isinstance(payload, dict):
        intent = payload.get("intent")
        if isinstance(intent, str) and intent.lower() in _INTENTS:
            return ToolIntentDecision(
                intent=cast(ToolIntent, intent.lower()),
                needs_retrieval=payload.get("needs_retrieval") is True,
            )
    return ToolIntentDecision()


def parse_tool_intent(response: str) -> ToolIntent:
    return parse_tool_intent_decision(response).intent


class ToolIntentRouter:
    """Run one tool-free classification turn on the active chat backend."""

    def classify(self, backend: Any, messages: list[ChatMessage]) -> ToolIntentDecision:
        conversation = [
            message
            for message in messages
            if message.role in {"user", "assistant"} and message.content
        ][-8:]
        classifier_messages = [
            ChatMessage(role="system", content=TOOL_INTENT_PROMPT),
            *conversation,
        ]
        completed_text: str | None = None
        config = ModelRequestConfig(max_tokens=128, temperature=0.0, top_p=1.0)

        for event in backend.stream_model_turn(classifier_messages, [], config):
            if not isinstance(event, ModelEvent):
                raise TypeError("Intent classifier backend returned an invalid event")
            if event.kind == "turn_complete" and event.turn is not None:
                completed_text = event.turn.text

        if completed_text is None:
            raise RuntimeError("Intent classifier backend did not complete its turn")
        return parse_tool_intent_decision(completed_text)
