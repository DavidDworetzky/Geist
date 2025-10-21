import requests
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
import subprocess
import os
import signal
import psutil
import json
import logging
from utils.logging import log_function_call
from agents.architectures.llama.llama_mlx import LlamaMLX
from agents.architectures.llama.llama_transformers import LlamaTransformer
from agents.models.agent_completion import LlamaCompletion
import torch
from typing import Optional, List, Dict, Any
from agents.architectures.sesame.generator import load_csm_1b
import torchaudio
import os
import time
from transformers import pipeline

WORLD_TICK_PROMPT = f"""You are a world class executive. Your plans are plans are direct, and detailed only if necessary. 
Given what you know about the world today, and the main task that you need to complete, consider if there are any additional facts that you should add to the list of things you consider. 
Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler statements."""

TASK_TICK_PROMPT = f"You are a focused individual. Given the main task that you wish to complete, and current working subtasks, create a specific list of actionable tasks that will complete your problem. Delimit these as plain english separated by the | character. Do not use function calls yet - only plain english."

FUNCTION_CALL_JSON = """
{
    "class" : "class_name",
    "function": "function_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}
"""

EXECUTION_TICK_PROMPT = f"You are given a list of tasks and list of function calls that you can make. Given the state of the world, and classes available to you - formulate a function call that will help you complete your task. You should formulate the function call as {FUNCTION_CALL_JSON}. Only call functions that are listed in our adapter list."


SYSTEM_PROMPT = "You are an agent looking to complete tasks for individuals. You will be given context about the world, the task and functions you can call. Take the most direct and thorough way of satisfying these constraints."

MAX_BATCH_SIZE = 1000

class LlamaAgent(BaseAgent):
    def __init__(self, agent_context, ckpt_dir, as_subprocess=False, pre_initialize_model=True):
        # call super constructor
        super().__init__(agent_context, as_subprocess)

        self.logger = logging.getLogger(__name__)
        if pre_initialize_model:
            if torch.backends.mps.is_available():
                self.logger.info("Using MPS (Apple Silicon) device - initializing LlamaMLX")
                self.llama = LlamaMLX(max_new_tokens=agent_context.settings.max_tokens)
            else:
                self.logger.info("Using CPU/CUDA device - initializing LlamaTransformer")
                self.llama = LlamaTransformer(max_new_tokens=agent_context.settings.max_tokens)

    def phase_out(self):
        self._agent_context._save()
        self.terminate_subprocess()

    def phase_in(self):
        self.initialize()
    
    def _complete_llama_sequence(self, prompt:str, system_prompt:str, max_tokens:int = None, top_p:float = None, temperature:float = None, streaming:bool = False):
        if not self.llama:
            if torch.backends.mps.is_available():
                self.llama = LlamaMLX(max_new_tokens=self._agent_context.settings.max_tokens, top_p = top_p if top_p else 0.95, temperature = temperature if temperature else 0.7)
            else:
                self.llama = LlamaTransformer(max_new_tokens=self._agent_context.settings.max_tokens)
        llama_completion = self.llama.complete(
            system_prompt=system_prompt,
            user_prompt=prompt,
            streaming=streaming
        )
        self.logger.info(f"Llama completion: {llama_completion}")
        return LlamaCompletion.from_dict(llama_completion)


    def _is_valid_function_json(self, function_json:str):
        try:
            function_json = function_json.replace('\n', '')
            # Attempt to parse the JSON string
            parsed_json = json.loads(function_json)
            # Check if the required keys are present
            required_keys = ["function", "parameters", "class"]
            if all(key in parsed_json for key in required_keys):
                # Check if parameters have the correct structure
                if isinstance(parsed_json["parameters"], dict) :
                    return True
            return False
        except json.JSONDecodeError:
            # Return False if JSON is invalid
            return False        
        

    def _take_json_and_call_function(self, function_json:str):
        '''
        takes a json definition of a function call and uses it to call one of our function adapters.
        '''
        if not self._is_valid_function_json(function_json):
            raise Exception(f"invalid function call json: {function_json}")
    
        json_data = json.loads(function_json)
    
        # now, find our adapter class and call the relevant function on it.
        class_name = json_data["class"]
        adapter_class = next((wrapper for wrapper in self._agent_context.initialized_classes if wrapper.name == class_name), None)

        if not adapter_class:
            raise Exception(f"No adapter class matching{class_name}")
    
        adapter_class = adapter_class.instance
        # now, call the relevant function on the adapter_class through reflection
        function_to_call = getattr(adapter_class, json_data["function"])
        parameters = json_data["parameters"]
    
        # Execute the function with the provided parameters
        result = function_to_call(**parameters)
    
        return result
        

    def _aggregated_context(self, world_context : bool, task_context : bool, execution_context: bool):
        #get aggregated context for world, task and execution context if requested
        context_string = ""
        if world_context:
            context_string += "WORLD_CONTEXT:" + "\n".join(self._agent_context.world_context)
        if task_context:
            context_string += "TASK_CONTEXT:" + "\n".join(self._agent_context.task_context)
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + "\n".join(self._agent_context.execution_context)
        return context_string

    def stream_complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
        return self.complete_text(prompt, max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty, stop, echo, best_of, prompt_tokens, response_format, system_prompt, chat_id, streaming=True)
        
    def complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None, streaming:bool = False):
        #set defaults for agent settings based off of settings values. If undefined,\
        max_tokens = self._agent_context.settings.max_tokens if self._agent_context.settings.max_tokens and not max_tokens else 16
        n = self._agent_context.settings.n if self._agent_context.settings.n and not n else 1
        temperature = self._agent_context.settings.temperature if self._agent_context.settings.temperature and not temperature else 1.0
        top_p = self._agent_context.settings.top_p if self._agent_context.settings.top_p and not top_p else 1
        frequency_penalty = self._agent_context.settings.frequency_penalty if self._agent_context.settings.frequency_penalty and not frequency_penalty else 0
        presence_penalty = self._agent_context.settings.presence_penalty if self._agent_context.settings.presence_penalty and not presence_penalty else 0
        
        if not system_prompt:
            system_prompt = SYSTEM_PROMPT

        completion = self._complete_llama_sequence(prompt = prompt, max_tokens = max_tokens if max_tokens else None, system_prompt=system_prompt, top_p = top_p if top_p else None, temperature = temperature if temperature else None)
        ai_message = next((gen.content for gen in completion.messages if gen.role == 'assistant'), None)
        chat_history = self._agent_context._add_to_chat_history(user_message=prompt, ai_message=ai_message, chat_id=chat_id)

        completion.chat_id = chat_history.chat_session_id

        return completion
        
    def initialize(self, task:str = None):
        #push task onto our stack for this agent.
        if task:
            self._agent_context.task_context.append(task)
        if self.as_subprocess:
            logging.info("Initializing agent with subprocess.")
            # Create a subprocess that runs one tick every second
            process = subprocess.Popen(['python3', '-u', 'tick.py'], stdout=subprocess.PIPE)

            # Set the subprocess ID in our agent context
            self._agent_context.subprocess_id = process.pid
        else:
            logging.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
            return None

    def terminate_subprocess(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self._agent_context.subprocess_id

        # Check if the subprocess ID is set
        if subprocess_id:
            # Send a terminate signal to the subprocess
            os.kill(subprocess_id, signal.SIGTERM)
            self._agent_context.subprocess_id = None
        else:
            raise Exception("No subprocess ID set in agent context.")
    
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
            result = result[0]
            retries = 0

            while not self._is_valid_function_json(result) and retries < 3:
                retries += 1
                result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
                result = self._transform_completions(result)
                result = result[0]

            if not self._is_valid_function_json(result):
                logging.error(f'Invaild result for function call is: {result}')
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
        self.tick_world()
        self._pop_and_add_execution_tasks()
        self._clear_execution_tasks()

    def is_subprocess_running(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self.agent_context.subprocess_id
        # check if subprocess is running and return value (by using psutil)
        try:
            p = psutil.Process(subprocess_id)
            return p.is_running()
        except psutil.NoSuchProcess:
            return False

    def _transform_completions(self, completion):
        try:
            choices_list = completion['choices']
            transformed_content = list(map(lambda x: x['message']['content'], choices_list))
            return transformed_content
        except Exception as e:
            logging.error(f"contents of completion, failed to destructure: {completion}, exception {e}")
            raise Exception(f"completion failed to destructure: {completion}. Format interop failure. Is your LLM protocol returning the correct format?")

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

    def complete_audio(self, audio_file: str, max_tokens: int = 16, n: int = 1,
                      stop: Optional[List[str]] = None, temperature: float = 1.0,
                      top_p: float = 1, frequency_penalty: float = 0,
                      presence_penalty: float = 0, echo: bool = False,
                      best_of: Optional[int] = None, prompt_tokens: Optional[int] = None,
                      response_format: str = "text", system_prompt: Optional[str] = None,
                      chat_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audio input and generate a response using the Sesame CSM voice model.

        Args:
            audio_file: Path to the audio file to process
            max_tokens: Maximum number of tokens to generate
            n: Number of completions to generate
            stop: Sequences where the API will stop generating further tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            echo: Whether to echo the prompt
            best_of: Number of completions to generate and return the best one
            prompt_tokens: Number of tokens in the prompt
            response_format: Format of the response ("text" or "audio")
            system_prompt: System prompt to use
            chat_id: ID of the chat

        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            # Initialize the voice generator if not already done
            if not hasattr(self, 'voice_generator') or self.voice_generator is None:
                self.voice_generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
                logger = logging.getLogger(__name__)
                logger.info("Initialized Sesame CSM voice generator")

            audio_tensor, sample_rate = torchaudio.load(audio_file)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=self.voice_generator.sample_rate
            )

            transcribe_model_id = "facebook/mms-1b-all"
            #assume english transcription until lparameterization
            transcribe_target_lang = "en"

            pipe = pipeline(model=transcribe_model_id, model_kwargs={"target_lang": transcribe_target_lang, "ignore_mismatched_sizes": True})
            transcribed_text = pipe(audio_tensor)
            
            # Generate a text response using the LLM
            text_response = self.complete_text(
                prompt=transcribed_text,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                echo=echo,
                best_of=best_of,
                prompt_tokens=prompt_tokens,
                response_format="text",
                system_prompt=system_prompt,
                chat_id=chat_id
            )

            # Extract the text from the LLM response
            if isinstance(text_response, dict) and "choices" in text_response:
                response_text = text_response["choices"][0]["text"]
            else:
                response_text = str(text_response)
            
            # Generate audio response using the Sesame CSM model
            audio_response = self.voice_generator.generate(
                text=response_text,
                speaker=0,  # Default speaker
                context=[],  # In a real implementation, you might want to include context
                max_audio_length_ms=10000,  # Default max audio length
            )

            # Save the audio response to a file
            output_dir = os.path.join(self._agent_context.settings.output_dir, "audio_responses")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"response_{chat_id or 'unknown'}_{int(time.time())}.wav")
            torchaudio.save(output_file, audio_response.unsqueeze(0).cpu(), self.voice_generator.sample_rate)

            return {
                "text": response_text,
                "audio_file": output_file,
                "sample_rate": self.voice_generator.sample_rate,
                "chat_id": chat_id
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in complete_audio: {str(e)}")
            return {"error": str(e)}

    def connect_realtime_audio(self):
        """
        Connects to a real-time audio stream (Not Implemented).

        Raises:
            NotImplementedError: This method is not yet implemented for LlamaAgent.
        """
        self.logger.warning("connect_realtime_audio is not implemented for LlamaAgent.")
        raise NotImplementedError("Real-time audio connection is not supported by this agent yet.")
