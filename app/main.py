import dataclasses
import json
import logging
import os
from typing import Any, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from adapters.mms_adapter import MMSAdapter
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.agent_type import AgentType

# Initialize agent architecture registry
from agents.architectures.registry import register_all_runners
from agents.factory import AgentFactory
from agents.models.agent_completion import AgentCompletion
from agents.models.tool_calling import ModelRequestConfig, ToolContext
from agents.online_agent import OnlineAgent
from agents.prompt.prompt import AGENT_PROMPTS, TOOL_USE_PROMPT
from app.api.v1.endpoints.files import router as files_router
from app.api.v1.endpoints.jobs import router as jobs_router
from app.api.v1.endpoints.memory import router as memory_router
from app.api.v1.endpoints.models import router as models_router
from app.api.v1.endpoints.providers import router as providers_router
from app.api.v1.endpoints.user_settings import router as user_settings_router
from app.api.v1.endpoints.voice import router as voice_router
from app.api.v1.endpoints.workflows import router as workflow_router
from app.environment import load_environment_dictionary
from app.models.completion import CompleteTextParams, InitializeAgentParams
from app.models.database.agent_preset import AgentPreset
from app.models.database.chat_session import (
    get_all_chat_history,
    get_chat_history,
    get_paginated_chat_history,
    get_paginated_chat_sessions,
)
from app.models.database.database import SessionLocal
from app.models.database.geist_user import get_default_user
from app.models.database.memory import MemoryFolder
from app.models.user_settings import AgentFactoryConfig
from app.services.chat_orchestrator import ChatOrchestrator, RunControlRegistry
from app.services.job_queue import start_worker, stop_worker
from app.services.memory_context import build_memory_context
from app.services.memory_scheduler import MEMORY_JOB_KIND  # noqa: F401
from app.services.memory_service import get_chat_memory_settings
from app.services.model_downloads import DOWNLOAD_JOB_KIND  # noqa: F401  (registers handler)
from app.services.tool_registry import build_default_tool_registry
from app.services.user_settings_service import UserSettingsService


# Register all available runners at startup
register_all_runners()

DEFAULT_PROMPT = AGENT_PROMPTS["default"]

load_dotenv()
DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_LOCAL_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
openai_key = os.getenv("OPENAI_API_KEY")
enhanced_logging = (os.getenv("ENHANCED_LOGGING") or "").strip().lower() in ("true", "1", "yes")

# in memory agent cache
agent_cache = {
    AgentType.LLAMA: None,
    AgentType.GPT4AGENT: None,
    AgentType.HTTPAGENT: None,
    AgentType.LOCALAGENT: None,
}

# mapping from public AgentType values to the agent factory's "local"/"online" types
AGENT_TYPE_TO_FACTORY_TYPE = {
    AgentType.LLAMA: "local",
    AgentType.LOCALAGENT: "local",
    AgentType.GPT4AGENT: "online",
    AgentType.HTTPAGENT: "online",
}

# constants
api_version = 1.0
default_agent_type = AgentType.LLAMA
run_controls = RunControlRegistry()
chat_orchestrator = ChatOrchestrator(
    build_default_tool_registry(),
    run_controls=run_controls,
)

if enhanced_logging:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_envs() -> dict[str, str]:
    return load_environment_dictionary()


def get_or_create_agent(agent_type: AgentType):
    if agent_cache[agent_type] is None:
        agent_cache[agent_type] = agent_mappings[agent_type]()
        logger.info(f"Created new {agent_type} agent")
    return agent_cache[agent_type]


def completion_to_response(completion) -> AgentCompletion:
    if isinstance(completion, AgentCompletion):
        return completion
    if completion:
        return AgentCompletion.from_completion(completion)
    raise HTTPException(status_code=500, detail="Failed to generate completions.")


def sse_event(event: str, payload: Any) -> str:
    if dataclasses.is_dataclass(payload):
        payload = dataclasses.asdict(cast(Any, payload))
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def chunk_completion_text(text: str, chunk_size: int = 160):
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def resolve_agent_type(value) -> AgentType:
    if value is None:
        return default_agent_type
    if isinstance(value, AgentType):
        return value
    return AgentType[str(value).upper()]


def model_request_config(params: CompleteTextParams) -> ModelRequestConfig:
    return ModelRequestConfig(
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        frequency_penalty=params.frequency_penalty,
        presence_penalty=params.presence_penalty,
        stop=params.stop,
    )


def chat_system_prompt(enable_tools: bool, memory_context: str = "") -> str:
    sections = [DEFAULT_PROMPT]
    if enable_tools:
        sections.append(TOOL_USE_PROMPT)
    if memory_context:
        sections.append(memory_context)
    return "\n\n".join(sections)


def resolved_memory_settings(
    params: CompleteTextParams,
    chat_id: int | None,
    user_id: int,
) -> tuple[bool, str, int | None]:
    if chat_id is not None:
        current = get_chat_memory_settings(user_id, chat_id)
        if current is not None:
            return (
                bool(current["memory_enabled"]),
                str(current["memory_mode"]),
                current["folder_id"],
            )
    folder_id = params.folder_id
    memory_mode = params.memory_mode if params.memory_mode in {"public", "private"} else "public"
    if folder_id is not None:
        with SessionLocal() as session:
            owned_folder = (
                session.query(MemoryFolder)
                .filter(
                    MemoryFolder.folder_id == folder_id,
                    MemoryFolder.user_id == user_id,
                )
                .first()
            )
        if owned_folder is None:
            raise HTTPException(status_code=422, detail="Memory folder not found")
        memory_mode = "private"
    return params.memory_enabled, memory_mode, folder_id


def run_chat_completion(
    params: CompleteTextParams,
    chat_id: int | None = None,
    agent=None,
) -> AgentCompletion:
    active_agent = agent or get_active_agent(resolve_agent_type(params.agent_type))
    user_id = int(get_default_user().user_id)
    memory_enabled, memory_mode, folder_id = resolved_memory_settings(params, chat_id, user_id)
    memory_context = build_memory_context(
        user_id,
        params.prompt,
        chat_session_id=chat_id,
        memory_enabled=memory_enabled,
        memory_mode=memory_mode,
        folder_id=folder_id,
    )
    if not hasattr(active_agent, "stream_model_turn"):
        completion = active_agent.complete_text(
            prompt=params.prompt,
            max_tokens=params.max_tokens,
            n=params.n,
            stop=params.stop,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            echo=params.echo,
            best_of=params.best_of,
            prompt_tokens=params.prompt_tokens,
            response_format=params.response_format,
            system_prompt=DEFAULT_PROMPT,
            chat_id=chat_id,
        )
        return completion_to_response(completion)
    return chat_orchestrator.complete(
        backend=active_agent,
        prompt=params.prompt,
        user_id=user_id,
        chat_id=chat_id,
        config=model_request_config(params),
        system_prompt=chat_system_prompt(params.enable_tools, memory_context),
        enable_tools=params.enable_tools,
        memory_enabled=memory_enabled,
        memory_mode=memory_mode,
        folder_id=folder_id,
    )


def stream_chat_completion(params: CompleteTextParams, chat_id: int | None = None):
    try:
        agent = get_active_agent(resolve_agent_type(params.agent_type))
        if hasattr(agent, "stream_model_turn"):
            user_id = int(get_default_user().user_id)
            memory_enabled, memory_mode, folder_id = resolved_memory_settings(
                params, chat_id, user_id
            )
            memory_context = build_memory_context(
                user_id,
                params.prompt,
                chat_session_id=chat_id,
                memory_enabled=memory_enabled,
                memory_mode=memory_mode,
                folder_id=folder_id,
            )
            for event in chat_orchestrator.stream(
                backend=agent,
                prompt=params.prompt,
                user_id=user_id,
                chat_id=chat_id,
                config=model_request_config(params),
                system_prompt=chat_system_prompt(params.enable_tools, memory_context),
                enable_tools=params.enable_tools,
                memory_enabled=memory_enabled,
                memory_mode=memory_mode,
                folder_id=folder_id,
            ):
                yield sse_event(event.event, event.payload)
            return

        # Legacy agents retain text-only behavior. Generate once and adapt the
        # authoritative completion into SSE chunks; consuming a legacy text stream
        # and then calling complete_text again can duplicate model work and database
        # persistence.
        completion = agent.complete_text(
            prompt=params.prompt,
            max_tokens=params.max_tokens,
            n=params.n,
            stop=params.stop,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            echo=params.echo,
            best_of=params.best_of,
            prompt_tokens=params.prompt_tokens,
            response_format=params.response_format,
            system_prompt=DEFAULT_PROMPT,
            chat_id=chat_id,
        )
        completion_object = completion_to_response(completion)
        for chunk in chunk_completion_text(
            completion_object.message[0] if completion_object.message else ""
        ):
            yield sse_event("delta", {"text": chunk})

        yield sse_event("final", completion_object)
        yield sse_event(
            "done",
            {"run_id": completion_object.run_id, "chat_id": completion_object.chat_id},
        )
    except Exception:
        logger.exception("Chat stream failed before a terminal event")
        yield sse_event(
            "error",
            {
                "code": "chat_backend_error",
                "message": (
                    "Chat backend failed to start. Check the configured model, "
                    "local weights, and required credentials."
                ),
                "chat_id": chat_id,
            },
        )
        yield sse_event("done", {"run_id": None, "chat_id": chat_id})


# App factory function
def create_app():
    app = FastAPI()

    # agent routes, for agentic flows.
    agent_router = APIRouter()
    # adapter routes, includes routes that address direct adapter calls.
    adapter_router = APIRouter()

    # Agent routes using agent_router
    @agent_router.post("/complete_text_new")
    def complete_text_new_agents(params: CompleteTextParams) -> AgentCompletion:
        """Complete text using new agent architecture (LocalAgent/OnlineAgent)."""
        from app.models.user_settings import AgentConfigRequest

        # Create agent config overrides from params
        requested_agent_type = resolve_agent_type(params.agent_type)
        overrides = AgentConfigRequest(
            agent_type=AGENT_TYPE_TO_FACTORY_TYPE.get(requested_agent_type),
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
        )

        # Get default agent context
        agent_context = get_default_agent_context()

        # Create agent using user settings
        agent = UserSettingsService.create_agent_from_default_user(agent_context, overrides)

        return run_chat_completion(params, agent=agent)

    @agent_router.post("/complete_text")
    def complete_text_endpoint(params: CompleteTextParams) -> AgentCompletion:
        return run_chat_completion(params)

    @agent_router.post("/complete_text_stream")
    async def complete_text_stream_endpoint(params: CompleteTextParams):
        return StreamingResponse(
            stream_chat_completion(params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @agent_router.post("/complete_text_stream/{session_id}")
    async def update_chat_session_and_stream_complete_text(
        params: CompleteTextParams, session_id: int
    ):
        return StreamingResponse(
            stream_chat_completion(params, chat_id=session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @agent_router.post("/complete_text/{session_id}")
    def update_chat_session_and_complete_text(params: CompleteTextParams, session_id: int):
        return run_chat_completion(params, chat_id=session_id)

    @agent_router.post("/runs/{run_id}/cancel")
    def cancel_chat_run(run_id: str):
        return {"run_id": run_id, "cancelled": run_controls.cancel(run_id)}

    @agent_router.get("/chat_history/{session_id}")
    async def get_chat_history_endpoint(session_id: int):
        return get_chat_history(session_id)

    @agent_router.get("/chat_history/{session_id}/paginated")
    async def get_paginated_history_endpoint(session_id: int, page: int = 1, page_size: int = 20):
        chat_history = get_paginated_chat_history(session_id, page, page_size)
        return chat_history

    @agent_router.get("/chat_sessions")
    async def get_chat_sessions():
        chat_sessions = get_all_chat_history()
        return chat_sessions

    @agent_router.get("/chat_sessions/paginated")
    async def get_paginated_sessions_endpoint(page: int = 1, page_size: int = 20):
        chat_sessions = get_paginated_chat_sessions(page, page_size)
        return chat_sessions

    @agent_router.get("/tools")
    async def get_chat_tool_catalog():
        user = get_default_user()
        enabled_names = {
            tool.name
            for tool in chat_orchestrator.registry.definitions_for_context(
                ToolContext(user_id=user.user_id, chat_id=None, run_id="catalog")
            )
        }
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters_schema(),
                    "enabled": tool.name in enabled_names,
                    "enabled_by_default": tool.enabled_by_default,
                    "requires_approval": tool.requires_approval,
                    "side_effect": tool.side_effect,
                    "source_adapter": tool.source_adapter,
                }
                for tool in chat_orchestrator.registry.catalog()
            ]
        }

    @agent_router.post("/initialize_task_and_tick")
    async def initialize_and_tick_agent(task_prompt: InitializeAgentParams):
        agent_type = agent_type = (
            AgentType[task_prompt.agent_type.upper()]
            if task_prompt.agent_type
            else default_agent_type
        )
        agent = get_active_agent(agent_type)

        agent.initialize(task_prompt.prompt)
        agent.tick()
        state_snapshot = agent.state()
        return state_snapshot

    @agent_router.post("/phase_out")
    async def phase_out_agent(agent_type: str):
        agent_to_phase = _get_cached_agent_or_404(agent_type)
        agent_to_phase.phase_out()

    @agent_router.post("/phase_in")
    async def phase_in_agent(agent_type: str):
        agent_to_phase = _get_cached_agent_or_404(agent_type)
        agent_to_phase.phase_in()

    # Adapter routes using adapter_router
    @adapter_router.post("/speech_to_text")
    async def create_upload_file(
        file: UploadFile = File(...), adapter: MMSAdapter = Depends(get_speech_to_text_client)
    ):
        return adapter.transcribe(file)

    # Register routers
    app.include_router(agent_router, prefix="/agent")
    app.include_router(adapter_router, prefix="/adapter")
    app.include_router(workflow_router, prefix="/api/v1/workflows", tags=["workflows"])
    app.include_router(files_router, prefix="/api/v1/files", tags=["files"])
    app.include_router(user_settings_router, prefix="/api/v1/user-settings", tags=["user-settings"])
    app.include_router(voice_router, prefix="/api/v1/voice", tags=["voice"])
    app.include_router(models_router, prefix="/api/v1/models", tags=["models"])
    app.include_router(providers_router, prefix="/api/v1/providers", tags=["providers"])
    app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["jobs"])
    app.include_router(memory_router, prefix="/api/v1/memory", tags=["memory"])

    @app.on_event("startup")
    def start_job_worker():
        start_worker()

    @app.on_event("shutdown")
    def stop_job_worker():
        stop_worker()

    @app.get("/")
    def version():
        return {"Version": f"{api_version}"}

    return app


def _parse_agent_type(agent_type: str) -> AgentType:
    try:
        return AgentType[agent_type.upper()]
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Unknown agent type: {agent_type}") from error


def _get_cached_agent_or_404(agent_type: str):
    parsed_type = _parse_agent_type(agent_type)
    agent = agent_cache.get(parsed_type)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No active agent of type {parsed_type.value}")
    return agent


def get_gpt4_client():
    """
    Legacy GPT4AGENT type: an OnlineAgent pinned to the OpenAI gpt-4 endpoint.
    """
    agent_context = get_default_agent_context()
    return OnlineAgent(
        agent_context=agent_context, base_url=DEFAULT_API_URL, model="gpt-4", api_key=openai_key
    )


def get_llama_agent():
    """
    Legacy LLAMA type: a LocalAgent running the default local model.
    """
    return get_local_agent()


def get_local_agent():
    agent_context = get_default_agent_context()
    # Get user settings to determine local model configuration
    settings = UserSettingsService.get_default_user_settings()
    factory_config = AgentFactoryConfig.from_user_settings(settings)
    model_id = settings.default_local_model or DEFAULT_LOCAL_MODEL
    return AgentFactory.create_agent(
        agent_type="local",
        agent_context=agent_context,
        model=model_id,
        generation_config=factory_config.generation_config,
    )


def get_online_agent():
    agent_context = get_default_agent_context()
    # Get user settings to determine provider-specific configuration
    settings = UserSettingsService.get_default_user_settings()
    factory_config = AgentFactoryConfig.from_user_settings(settings)

    return AgentFactory.create_agent(
        agent_type="online",
        agent_context=agent_context,
        # Preserve None so AgentFactory can infer the catalog endpoint and can
        # fail safely when a self-hosted model has no configured server.
        endpoint=factory_config.endpoint,
        model=factory_config.model,
        backup_providers=[provider.model_dump() for provider in factory_config.backup_providers],
        generation_config=factory_config.generation_config,
    )


agent_mappings = {
    AgentType.GPT4AGENT: get_gpt4_client,
    AgentType.LLAMA: get_llama_agent,
    AgentType.LOCALAGENT: get_local_agent,
    AgentType.HTTPAGENT: get_online_agent,
}


def get_active_agent(type: AgentType):
    return get_or_create_agent(type)


def get_default_agent_context():
    """
    Gets an agent context matching "Default Context" from the database
    """
    with SessionLocal() as session:
        # Query for the agent preset with name "Default Context"
        default_preset = (
            session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first()
        )
        logging.info(f"Default agent preset: {default_preset}")

        if not default_preset:
            raise ValueError("Default Context preset not found in the database.")

        agent_settings = AgentSettings(
            name=default_preset.name or "Default Preset",
            version=default_preset.version or "1.0",
            description=default_preset.description or "",
            max_tokens=int(default_preset.max_tokens or 16),
            n=int(default_preset.n or 1),
            temperature=float(
                default_preset.temperature if default_preset.temperature is not None else 1.0
            ),
            top_p=float(default_preset.top_p if default_preset.top_p is not None else 1.0),
            frequency_penalty=float(default_preset.frequency_penalty or 0),
            presence_penalty=float(default_preset.presence_penalty or 0),
            interactive_only=bool(default_preset.interactive_only),
            include_world_processing=bool(default_preset.process_world),
        )
        # Create an agent context with the found preset
        context = AgentContext(settings=agent_settings, envs=get_envs())
        return context


def get_speech_to_text_client():
    return MMSAdapter()


# Initialize and run the app
app = create_app()
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,  # 1MB (1024 * 1024 bytes)
    )
