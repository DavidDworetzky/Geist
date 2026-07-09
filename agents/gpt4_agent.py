import requests
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
from typing import Optional, Iterator
import subprocess
import os
import signal
import psutil
import logging
from utils.logging import log_function_call
from agents.models.gpt4_completion import Gpt4Completion
from app.models.database.chat_session import get_chat_history
from agents.response_utils import (
    AgentResponseMixin,
    WORLD_TICK_PROMPT,
    TASK_TICK_PROMPT,
    EXECUTION_TICK_PROMPT,
    build_chat_messages,
)


class GPT4Agent(AgentResponseMixin, BaseAgent):
    def __init__(self, api_key, agent_context, as_subprocess=False):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # call super constructor
        super().__init__(agent_context, as_subprocess)

    def phase_out(self):
        self.save_state_snapshot(reason="phase_out")
        self._agent_context._save()
        self.terminate_subprocess()

    def phase_in(self):
        self.initialize()
        self.restore_state_snapshot()

    def _complete_text(self, url:str, payload:any, headers:any) -> Gpt4Completion:
        response = requests.post(url, json=payload, headers=headers)
        response_content:Gpt4Completion = response.json() if response.status_code == 200 else None
        if not response_content:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        return response_content

    def complete_audio(self, audio_file, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
        #use open ai realtime api to get an audio response
        url = "https://api.openai.com/v1/audio/transcriptions"
        payload = {
            "file": audio_file,
            "model": "whisper-1",
            "response_format": "text",
            "temperature": temperature,
            "top_p": top_p,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response_content:Gpt4Completion = response.json() if response.status_code == 200 else None
        if not response_content:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        return response_content

    def stream_complete_text(self, prompt: str, max_tokens: int = 16, n: int = 1, stop: Optional[str] = None, temperature: float = 1, top_p: float = 1, frequency_penalty: float = 0, presence_penalty: float = 0, echo: bool = False, best_of: Optional[int] = None, prompt_tokens: Optional[int] = None, response_format: str = "text", system_prompt: Optional[str] = None, chat_id: Optional[int] = None) -> Iterator[str]:
        return self.complete_text(prompt, max_tokens, n, stop, temperature, top_p, frequency_penalty, presence_penalty, echo, best_of, prompt_tokens, response_format, system_prompt, chat_id, streaming=True)

    def complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None, streaming:bool = False) -> Gpt4Completion:
        # Normalize parameters using shared utility
        params = self._normalize_params(max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty)

        # Build messages with chat history hydration
        history = None
        if chat_id is not None:
            try:
                history = get_chat_history(chat_id)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to hydrate chat history for chat_id={chat_id}: {e}")

        messages = build_chat_messages(prompt, system_prompt, history)

        payload = {
            "messages": messages,
            "model": "gpt-4",
            "max_tokens": params.max_tokens,
            "n": params.n,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "stream": streaming
        }

        if stop is not None:
            payload["stop"] = stop

        completion = self._complete_text(self.base_url, payload, self.headers)
        completion_object = Gpt4Completion.from_dict(completion)

        # Add to chat history using the common extraction method
        ai_message = completion_object.get_assistant_content()
        chat_history = self._agent_context._add_to_chat_history(user_message=prompt, ai_message=ai_message, chat_id=chat_id)
        completion_object.chat_id = chat_history.chat_session_id

        return completion_object
        
    def initialize(self, task:str = None):
        #push task onto our stack for this agent.
        if task:
            self._agent_context.task_context.append(task)
        if self.as_subprocess:
            logging.info("Initializing agent with subprocess.")
            try:
                # Create a subprocess that runs one tick every second
                process = subprocess.Popen(
                    ['python3', '-u', 'tick.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Set the subprocess ID in our agent context
                self._agent_context.subprocess_id = process.pid
            except Exception as e:
                logging.error(f"Failed to start subprocess: {e}")
                self._agent_context.subprocess_id = None
        else:
            logging.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
            return None

    def terminate_subprocess(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self._agent_context.subprocess_id

        # Check if the subprocess ID is set
        if subprocess_id:
            try:
                # Send a terminate signal to the subprocess
                os.kill(subprocess_id, signal.SIGTERM)
            except ProcessLookupError:
                logging.warning(f"Process {subprocess_id} not found, may have already terminated")
            except OSError as e:
                logging.error(f"Error terminating subprocess {subprocess_id}: {e}")
            finally:
                self._agent_context.subprocess_id = None
    
    @log_function_call
    def _pop_and_add_execution_tasks(self):
            if self._agent_context.task_context:
                task_to_execute = self._agent_context.task_context.pop(0)
                prompt = f"executing task: {task_to_execute}" + self._aggregated_context(world_context=True, task_context=True, execution_context=True)
                execution_result = self.complete_text(prompt=TASK_TICK_PROMPT + prompt)
                execution_result = self._transform_completions(execution_result)
                tasks_to_execute = [task.strip() for result in execution_result for task in result.split('|')]
                self._agent_context.execution_context = tasks_to_execute
                return True
            else:
                raise Exception("No tasks available in task context for execution.")
            
    @log_function_call
    def _clear_execution_tasks(self):
        results = []
        for task in self._agent_context.execution_context:
            context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
            result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
            result = self._transform_completions(result)
            #get first result as function call
            if not result:
                logging.error(f"Empty result for task: {task}")
                continue
            result = result[0]
            retries = 0

            while not self._is_valid_function_json(result) and retries < 3:
                retries += 1
                result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
                result = self._transform_completions(result)
                if not result:
                    logging.error(f"Empty result on retry {retries} for task: {task}")
                    continue
                result = result[0]

            if not self._is_valid_function_json(result):
                logging.error(f'Invalid result for function call is: {result}')
                raise Exception("Exceeded retries for valid function call JSON.")
            
            output = self._take_json_and_call_function(result)
            results.append(output)

        self._agent_context.execution_context = []
        return results
        

    @log_function_call
    def tick(self):
        logging.info("Agent Tick.")
        # Make one inference call to GPT-4 to advance world state reasoning, tasks and then execute.
        # reason about the world, then.
        # pop elements of agent context tasks for execution
        if self._agent_context.include_world_processing:
            self.tick_world()
        self._pop_and_add_execution_tasks()
        self._clear_execution_tasks()
        self.save_state_snapshot(reason="tick")

    def is_subprocess_running(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self._agent_context.subprocess_id
        if not subprocess_id:
            return False
        # check if subprocess is running and return value (by using psutil)
        try:
            p = psutil.Process(subprocess_id)
            return p.is_running()
        except psutil.NoSuchProcess:
            return False

    @log_function_call
    def tick_world(self):
        '''
        Advances the state of the world context, reasoning about the world state.
        '''
        #aggregate list of strings from world_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        input = WORLD_TICK_PROMPT + context_string
        result = self.complete_text(prompt=input)
        #split result and replace world context
        split_result = self._transform_completions(result)
        self._agent_context.world_context = split_result
        return split_result

    def tick_tasks(self):
        '''
        Advances the state of the task context, reasoning about how to complete the primary task.
        '''
        #aggregate list of strings from task_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=TASK_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        #split result and replace task context
        split_result = result.split("\n")
        self.agent_context.task_context = split_result
        return split_result
    
    
    def tick_execution(self):
        '''
        Advances the state of the execution context, reasoning about how to execute the primary task.
        '''
        #aggregate list of strings from execution_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=EXECUTION_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        #split result and replace execution context
        split_result = result.split("\n")
        self._agent_context.execution_context = split_result
        return split_result

    def connect_realtime_audio(self):
        """
        Connects to a real-time audio stream (Not Implemented for GPT4Agent).

        Raises:
            NotImplementedError: This method is not yet implemented for GPT4Agent.
        """
        raise NotImplementedError("Real-time audio connection is not supported by this agent yet.")
