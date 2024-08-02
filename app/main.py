#framework
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

load_dotenv()
openai_key = os.getenv("OPENAI_TOKEN")
enhanced_logging = os.getenv("ENHANCED_LOGGING")
enhanced_logging = json.loads(enhanced_logging.lower())

#constants
api_version = 0.1

if enhanced_logging:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

def get_envs() -> dict[str,str]:
    return LoadEnvironmentDictionary()

# App factory function
def create_app():
    app = FastAPI()

    #agent routes, for agentic flows.
    agent_router = APIRouter()
    #adapter routes, includes routes that address direct adapter calls.
    adapter_router = APIRouter()

    # Agent routes using agent_router
    @agent_router.post("/complete_text")
    async def complete_text_endpoint(params: CompleteTextParams):
        agent_type = AgentType[params.agent_type.upper()]
        #parse agent_type from str to AgentType

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
            response_format=params.response_format
        )

        if completions:
            return completions
        else:
            raise HTTPException(status_code=500, detail="Failed to generate completions.")

    @agent_router.post("/initialize_task_and_tick")
    async def initialize_and_tick_agent(task_prompt: InitializeAgentParams):
        agent_type = AgentType[task_prompt.agent_type.upper()]
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
    return LlamaAgent(agent_context = agent_context)

agent_mappings = { 
    AgentType.GPT4AGENT : get_gpt4_client,
    AgentType.LLAMA : get_llama_agent
}

def get_active_agent(type: AgentType):
    return agent_mappings[type]()

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
