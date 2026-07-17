"""Run Geist with a deterministic chat agent for browser end-to-end tests."""

import os
from typing import Any

import uvicorn

from agents.models.agent_completion import AgentCompletion
from initdb import main as initialize_database


class BrowserE2EAgent:
    """Exercise the production legacy-agent SSE adapter without loading a model."""

    def complete_text(self, **kwargs: Any) -> AgentCompletion:
        if kwargs["prompt"] == "Trigger backend failure":
            raise RuntimeError("browser e2e injected failure")
        return AgentCompletion(
            message=["E2E chat works."],
            id="e2e-completion",
            chat_id=101,
            run_id="e2e-run",
        )


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
