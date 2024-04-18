#framework
import sys
from typing import Optional
import logging
from app.models.completion import CompleteTextParams, InitializeAgentParams
from agents.gpt4_agent import GPT4Agent
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from adapters.mms_adapter import MMSAdapter
from dotenv import load_dotenv
from agents import agent_context
from agents.agent_settings import AgentSettings
import os
from app.models.database.database import Session
from app.models.database.agent_preset import AgentPreset  
from agents.agent_context import AgentContext  
from app.models.agent import Agent
import uvicorn

load_dotenv()
openai_key = os.getenv("OPENAI_TOKEN")

#constants
api_version = 0.1
enhanced_logging = False

if enhanced_logging:
    logging.basicConfig()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# App factory function
def create_app():
    app = FastAPI()

    @app.get('/')
    def version():
        return {"Version": f"{api_version}"}

    @app.post("/complete_text")
    async def complete_text_endpoint(params: CompleteTextParams, agent: GPT4Agent = Depends(get_gpt4_client)):
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

    @app.post("/speech_to_text")
    async def create_upload_file(file: UploadFile = File(...), adapter: MMSAdapter = Depends(get_speech_to_text_client)):
        return adapter.transcribe(file)

    @app.post("/initialize_task_and_tick")
    async def initialize_and_tick_agent(task_prompt: InitializeAgentParams, agent: GPT4Agent = Depends(get_gpt4_client)):
        agent.initialize(task_prompt)
        agent.tick()
        return agent.state()


    @app.post("/phase_out")
    async def phase_out_agent(agent_id: int):
        agent_to_phase = Agent.get_agent_by_id(agent_id)
        agent_to_phase.phase_out()

    @app.post("/phase_in")
    async def phase_in_agent(agent_id: int):
        agent_to_phase = Agent.get_agent_by_id(agent_id)
        agent_to_phase.phase_in()

    return app


def get_gpt4_client():
    agent_context = get_default_agent_context()
    api_key = openai_key
    return GPT4Agent(api_key=api_key, agent_context=agent_context)

def get_default_agent_context():
    '''
    Gets an agent context matching "Default Context" from the database
    '''
    session = Session
    try:
        # Query for the agent preset with name "Default Context"
        default_preset = session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first()
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
            interactive_only=default_preset.interactive_only
        )
        if default_preset:
            # Create an agent context with the found preset
            context = AgentContext(settings=agent_settings)
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
