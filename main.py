#framework
import sys
from typing import Optional
import logging
from app.models.completion import CompleteTextParams
from agents.gpt4_agent import GPT4Agent
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from adapters.mms_adapter import MMSAdapter
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_TOKEN")

#constants
api_version = 0.1
enhanced_logging = False

if enhanced_logging:
    logging.basicConfig()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


#initialization
app = FastAPI()

	
#routes
@app.get('/')
def version():
    return {"Version" : f"{api_version}"}

def get_gpt4_client():
    api_key = openai_key
    return GPT4Agent(api_key)

def get_speech_to_text_client():
    return MMSAdapter()


#basic text completion endpoint for our agent.
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
    
