import sys
from typing import Optional
import logging
from app.models.completion import CompleteTextParams, InitializeAgentParams
from agents.gpt4_agent import GPT4Agent
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, APIRouter
from adapters.mms_adapter import MMSAdapter
from dotenv import load_dotenv
from agents import agent_context
from agents.agent_settings import AgentSettings
import os
from app.models.database.database import Session
from app.models.database.agent_preset import AgentPreset  
from agents.agent_context import AgentContext  
from app.models.agent import Agent
from app.environment import LoadEnvironmentDictionary
import uvicorn
import json
from agents.llama_agent import LlamaAgent
from agents.agent_type import AgentType
from agents.models.agent_completion import AgentCompletion
from agents.prompt.prompt import AGENT_PROMPTS
from app.models.database.chat_session import get_chat_history, get_all_chat_history
from app.api.v1.endpoints.workflows import router as workflow_router

DEFAULT_PROMPT = AGENT_PROMPTS["default"]

load_dotenv()
openai_key = os.getenv("OPENAI_TOKEN")
enhanced_logging = os.getenv("ENHANCED_LOGGING")
enhanced_logging = json.loads(enhanced_logging.lower()) if enhanced_logging else False

#in memory agent cache
agent_cache = {
    AgentType.LLAMA : None,
    AgentType.GPT4AGENT : None
}

#constants
api_version = 1.0
default_agent_type = AgentType.LLAMA

if enhanced_logging:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def get_envs() -> dict[str,str]:
    return LoadEnvironmentDictionary()

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
    @agent_router.post("/complete_text")
    async def complete_text_endpoint(params: CompleteTextParams) -> AgentCompletion:
        #if params.agent_type is an AgentType, use it.
        if not isinstance(params.agent_type, AgentType):
            agent_type = AgentType[params.agent_type.upper()]
        else:
            agent_type = params.agent_type

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
        #if params.agent_type is an AgentType, use it.
        if not isinstance(params.agent_type, AgentType):
            agent_type = AgentType[params.agent_type.upper()]
        else:
            agent_type = params.agent_type

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
    async def get_chat_history(session_id: int):
        chat_history = get_chat_history(session_id)
        return chat_history
    
    @agent_router.get("/chat_sessions")
    async def get_chat_sessions():
        chat_sessions = get_all_chat_history()
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
    async def phase_out_agent(agent_id: int):
        agent_to_phase = Agent.get_agent_by_id(agent_id)
        agent_to_phase.phase_out()

    @agent_router.post("/phase_in")
    async def phase_in_agent(agent_id: int):
        agent_to_phase = Agent.get_agent_by_id(agent_id)
        agent_to_phase.phase_in()

    # Adapter routes using adapter_router
    @adapter_router.post("/speech_to_text")
    async def create_upload_file(file: UploadFile = File(...), adapter: MMSAdapter = Depends(get_speech_to_text_client)):
        return adapter.transcribe(file)

    # Register routers
    app.include_router(agent_router, prefix="/agent")
    app.include_router(adapter_router, prefix="/adapter")
    app.include_router(workflow_router, prefix="/api/v1/workflows", tags=["workflows"])

    @app.get('/')
    def version():
        return {"Version": f"{api_version}"}

    return app

def get_gpt4_client():
    agent_context = get_default_agent_context()
    api_key = openai_key
    return GPT4Agent(api_key=api_key, agent_context=agent_context)

def get_llama_agent():
    agent_context = get_default_agent_context()
    return LlamaAgent(agent_context = agent_context, ckpt_dir=None)

agent_mappings = { 
    AgentType.GPT4AGENT : get_gpt4_client,
    AgentType.LLAMA : get_llama_agent
}

def get_active_agent(type: AgentType):
    return get_or_create_agent(type)

def get_default_agent_context():
    '''
    Gets an agent context matching "Default Context" from the database
    '''
    session = Session
    try:
        # Query for the agent preset with name "Default Context"
        default_preset = session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first()
        logging.info(f"Default agent preset: {default_preset}")
        agent_settings = AgentSettings(
            name=default_preset.name,
            version=default_preset.version,
            description=default_preset.description,
            max_tokens=default_preset.max_tokens,
            n=default_preset.n,
            temperature=default_preset.temperature,
            top_p=default_preset.top_p,
            frequency_penalty=default_preset.frequency_penalty,
            presence_penalty=default_preset.presence_penalty,
            interactive_only=default_preset.interactive_only,
            include_world_processing= default_preset.process_world,
        )
        if default_preset:
            # Create an agent context with the found preset
            context = AgentContext(settings=agent_settings, envs=get_envs())
            return context
        else:
            raise ValueError("Default Context preset not found in the database.")
    finally:
        session.close()


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
