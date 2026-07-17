"""Run Geist with a deterministic chat agent for browser end-to-end tests."""

import os
from typing import Any

import uvicorn

from agents.models.tool_calling import ChatMessage, ModelEvent, ModelTurn
from initdb import main as initialize_database


class BrowserE2EAgent:
    """Exercise the production orchestrator without loading a model."""

    supports_native_tool_calling = False

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        _tools: list[Any],
        _config: Any,
    ):
        prompt = next(
            message.content or "" for message in reversed(messages) if message.role == "user"
        )
        if prompt == "Trigger backend failure":
            raise RuntimeError("browser e2e injected failure")
        if prompt == "Remember cobalt.":
            response = "I will remember cobalt."
        elif prompt == "What should you remember?":
            prior_user_messages = [
                message.content for message in messages[:-1] if message.role == "user"
            ]
            response = "cobalt" if "Remember cobalt." in prior_user_messages else "missing context"
        else:
            response = "E2E chat works."
        yield ModelEvent.text_delta(response)
        yield ModelEvent.turn_complete(ModelTurn(text=response, finish_reason="stop"))


def run_server() -> None:
    initialize_database()

    import app.main as geist_main

    def get_e2e_agent(_agent_type: Any) -> BrowserE2EAgent:
        return BrowserE2EAgent()

    geist_main.get_active_agent = get_e2e_agent
    uvicorn.run(
        geist_main.app,
        host=os.getenv("GEIST_E2E_BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("GEIST_E2E_BACKEND_PORT", "5100")),
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
