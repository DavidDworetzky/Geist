import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile

from adapters.mms_adapter import MMSAdapter
from agents.agent_type import AgentType

# Initialize agent architecture registry
from agents.architectures.registry import register_all_runners
from agents.local_agent import LocalAgent
from agents.models.agent_completion import AgentCompletion
from agents.online_agent import OnlineAgent
from agents.prompt.prompt import AGENT_PROMPTS
from app.api.v1.endpoints.files import router as files_router
from app.api.v1.endpoints.models import router as models_router
from app.api.v1.endpoints.user_settings import router as user_settings_router
from app.api.v1.endpoints.voice import router as voice_router
from app.api.v1.endpoints.workflows import router as workflow_router
from app.environment import load_environment_dictionary
from app.models.completion import CompleteTextParams, InitializeAgentParams
from app.models.database.chat_session import (
    get_all_chat_history,
    get_chat_history,
    get_paginated_chat_history,
    get_paginated_chat_sessions,
)
from app.models.user_settings import AgentFactoryConfig
from app.services.agent_context_provider import get_default_agent_context
from app.services.user_settings_service import UserSettingsService


# Register all available runners at startup
register_all_runners()

DEFAULT_PROMPT = AGENT_PROMPTS["default"]

load_dotenv()
DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_LOCAL_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
openai_key = os.getenv("OPENAI_API_KEY")
enhanced_logging = (os.getenv("ENHANCED_LOGGING") or "").strip().lower() in ("true", "1", "yes")

#in memory agent cache
agent_cache = {
    AgentType.LLAMA : None,
    AgentType.GPT4AGENT : None,
    AgentType.HTTPAGENT : None,
    AgentType.LOCALAGENT : None
}

#mapping from public AgentType values to the agent factory's "local"/"online" types
AGENT_TYPE_TO_FACTORY_TYPE = {
    AgentType.LLAMA: "local",
    AgentType.LOCALAGENT: "local",
    AgentType.GPT4AGENT: "online",
    AgentType.HTTPAGENT: "online",
}

#constants
api_version = 1.0
default_agent_type = AgentType.LLAMA

if enhanced_logging:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def get_envs() -> dict[str,str]:
    return load_environment_dictionary()

def _resolve_agent_type(agent_type: AgentType | str | None) -> AgentType:
    if agent_type is None:
        return default_agent_type
    if isinstance(agent_type, AgentType):
        return agent_type
    return AgentType[agent_type.upper()]

def get_or_create_agent(agent_type: AgentType):
    if agent_cache[agent_type] is None:
        agent_cache[agent_type] = agent_mappings[agent_type]()
        logger.info(f"Created new {agent_type} agent")
    return agent_cache[agent_type]

# App factory function
def create_app():
    app = FastAPI()

    #agent routes, for agentic flows.
    agent_router = APIRouter()
    #adapter routes, includes routes that address direct adapter calls.
    adapter_router = APIRouter()

    # Agent routes using agent_router
    @agent_router.post("/complete_text_new")
    async def complete_text_new_agents(params: CompleteTextParams) -> AgentCompletion:
        """Complete text using new agent architecture (LocalAgent/OnlineAgent)."""
        from app.models.user_settings import AgentConfigRequest

        # Create agent config overrides from params
        overrides = AgentConfigRequest(
            agent_type=AGENT_TYPE_TO_FACTORY_TYPE.get(params.agent_type) if params.agent_type else None,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty
        )

        # Get default agent context
        agent_context = get_default_agent_context()

        # Create agent using user settings
        agent = UserSettingsService.create_agent_from_default_user(agent_context, overrides)

        # Complete text
        completions = agent.complete_text(
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
            system_prompt=DEFAULT_PROMPT
        )

        if completions:
            completion_object = AgentCompletion.from_completion(completions)
            return completion_object
        else:
            raise HTTPException(status_code=500, detail="Failed to generate completions.")

    @agent_router.post("/complete_text")
    async def complete_text_endpoint(params: CompleteTextParams) -> AgentCompletion:
        agent_type = _resolve_agent_type(params.agent_type)

        agent = get_active_agent(agent_type)

        completions = agent.complete_text(
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
            system_prompt= DEFAULT_PROMPT
        )

        if completions:
            completion_object = AgentCompletion.from_completion(completions)
            return completion_object
        else:
            raise HTTPException(status_code=500, detail="Failed to generate completions.")

    @agent_router.post("/complete_text/{session_id}")
    async def update_chat_session_and_complete_text(params: CompleteTextParams, session_id: int):
        agent_type = _resolve_agent_type(params.agent_type)

        agent = get_active_agent(agent_type)

        completions = agent.complete_text(
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
            system_prompt= DEFAULT_PROMPT,
            chat_id=session_id
        )

        if completions:
            completion_object = AgentCompletion.from_completion(completions)
            return completion_object
        else:
            raise HTTPException(status_code=500, detail="Failed to generate completions.")

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

    @agent_router.post("/initialize_task_and_tick")
    async def initialize_and_tick_agent(task_prompt: InitializeAgentParams):
        agent_type = agent_type = AgentType[task_prompt.agent_type.upper()] if task_prompt.agent_type else default_agent_type
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
    async def create_upload_file(file: UploadFile = File(...), adapter: MMSAdapter = Depends(get_speech_to_text_client)):
        return adapter.transcribe(file)

    # Register routers
    app.include_router(agent_router, prefix="/agent")
    app.include_router(adapter_router, prefix="/adapter")
    app.include_router(workflow_router, prefix="/api/v1/workflows", tags=["workflows"])
    app.include_router(files_router, prefix="/api/v1/files", tags=["files"])
    app.include_router(user_settings_router, prefix="/api/v1/user-settings", tags=["user-settings"])
    app.include_router(voice_router, prefix="/api/v1/voice", tags=["voice"])
    app.include_router(models_router, prefix="/api/v1/models", tags=["models"])

    @app.get('/')
    def version():
        return {"Version": f"{api_version}"}

    return app

def _parse_agent_type(agent_type: str) -> AgentType:
    try:
        return AgentType[agent_type.upper()]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown agent type: {agent_type}") from None

def _get_cached_agent_or_404(agent_type: str):
    parsed_type = _parse_agent_type(agent_type)
    agent = agent_cache.get(parsed_type)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No active agent of type {parsed_type.value}")
    return agent

def get_gpt4_client():
    '''
    Legacy GPT4AGENT type: an OnlineAgent pinned to the OpenAI gpt-4 endpoint.
    '''
    agent_context = get_default_agent_context()
    return OnlineAgent(
        agent_context=agent_context,
        base_url=DEFAULT_API_URL,
        model="gpt-4",
        api_key=openai_key
    )

def get_llama_agent():
    '''
    Legacy LLAMA type: a LocalAgent running the default local model.
    '''
    return get_local_agent()

def get_local_agent():
    agent_context = get_default_agent_context()
    # Get user settings to determine local model configuration
    settings = UserSettingsService.get_default_user_settings()
    model_id = settings.default_local_model or DEFAULT_LOCAL_MODEL
    return LocalAgent(agent_context=agent_context, model_id=model_id, runner_type="mlx_llama")

def get_online_agent():
    agent_context = get_default_agent_context()
    # Get user settings to determine provider-specific configuration
    settings = UserSettingsService.get_default_user_settings()
    factory_config = AgentFactoryConfig.from_user_settings(settings)

    # Get provider-specific API key from environment
    provider = settings.default_online_provider
    api_key = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider == "grok":
        api_key = os.getenv("GROK_API_KEY")

    return OnlineAgent(
        agent_context=agent_context,
        base_url=factory_config.endpoint or DEFAULT_API_URL,
        model=factory_config.model,
        api_key=api_key
    )

agent_mappings = {
    AgentType.GPT4AGENT : get_gpt4_client,
    AgentType.LLAMA : get_llama_agent,
    AgentType.LOCALAGENT : get_local_agent,
    AgentType.HTTPAGENT : get_online_agent
}

def get_active_agent(type: AgentType):
    return get_or_create_agent(type)

def get_speech_to_text_client():
    return MMSAdapter()

# Initialize and run the app
app = create_app()
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000, # 1MB (1024 * 1024 bytes)
    )
