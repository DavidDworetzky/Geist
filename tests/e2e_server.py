"""Run Geist with a deterministic chat agent for browser end-to-end tests."""

import os
from typing import Any, Literal

import uvicorn

from adapters.search_adapter import SearchAdapter
from agents.models.tool_calling import ChatMessage, ModelEvent, ModelTurn
from initdb import main as initialize_database
from tests.streaming_probe import StreamingProbe


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
        if prompt == "Trigger failure after prose":
            from agents.architectures.chat_template_tools import ToolResponseStream

            parser = ToolResponseStream({"safe": "web.search"})
            yield ModelEvent.text_delta(parser.feed("Working on it. "))
            parser.feed("<tool_call>{bad}</tool_call>")
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

    from fastapi import HTTPException

    import app.main as geist_main
    from app.models.user_settings import UserSettingsUpdate
    from app.services.user_settings_service import UserSettingsService

    settings = UserSettingsService.get_default_workspace_settings()
    if (
        UserSettingsService.update_workspace_settings_by_id(
            settings.user_id,
            UserSettingsUpdate(default_agent_type="online"),
        )
        is None
    ):
        raise RuntimeError("browser E2E workspace settings were not initialized")

    probe = StreamingProbe()
    original_search = SearchAdapter.search

    def search_for_probe(self, search_term, max_results=5, recency=None):
        if probe.active and probe.scenario == "xml_tool":
            probe.search_calls.append({"query": search_term, "max_results": max_results})
            return [
                {
                    "title": "Fixture celebrity headline",
                    "url": "https://example.com/news",
                    "snippet": "Test search result",
                }
            ]
        return original_search(self, search_term, max_results=max_results, recency=recency)

    SearchAdapter.search = search_for_probe
    original_intent_router_enabled = geist_main.intent_router_enabled

    def get_e2e_agent(_agent_type: Any):
        return probe.agent() if probe.active else BrowserE2EAgent()

    geist_main.get_active_agent = get_e2e_agent
    geist_main.intent_router_enabled = lambda workspace_id: (
        False if probe.active else original_intent_router_enabled(workspace_id)
    )
    app = geist_main.create_app()

    @app.post("/api/e2e/streaming/start")
    def start_streaming_probe(scenario: Literal["text", "xml_tool"] = "text") -> dict[str, Any]:
        probe.start(scenario)
        return probe.state()

    @app.get("/api/e2e/streaming/state")
    def streaming_probe_state() -> dict[str, Any]:
        return probe.state()

    @app.post("/api/e2e/streaming/release/{stage}")
    def release_streaming_probe(stage: int) -> dict[str, Any]:
        if stage not in {1, 2}:
            raise HTTPException(status_code=422, detail="Expected stage 1 or 2")
        probe.gates[stage - 1].set()
        return probe.state()

    @app.post("/api/e2e/streaming/reset")
    def reset_streaming_probe() -> dict[str, Any]:
        probe.reset()
        return probe.state()

    if web_dir := os.getenv("GEIST_E2E_WEB_DIR"):
        # Register probe controls before the SPA catch-all, or it shadows them.
        app.state.web_dir = web_dir
        geist_main.install_spa(app, web_dir)

    uvicorn.run(
        app,
        host=os.getenv("GEIST_E2E_BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("GEIST_E2E_BACKEND_PORT", "5100")),
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
