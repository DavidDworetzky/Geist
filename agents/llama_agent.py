import requests
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
import subprocess
import os
import signal
import psutil
import logging
from utils.logging import log_function_call
from agents.architectures.llama.llama_mlx import LlamaMLX
from agents.architectures.llama.llama_transformers import LlamaTransformer
from agents.models.agent_completion import LlamaCompletion
import torch
from typing import Optional, List, Dict, Any
from agents.architectures.sesame.generator import load_csm_1b
import torchaudio
import time
from transformers import pipeline
from app.models.database.chat_session import get_chat_history
from agents.response_utils import (
    AgentResponseMixin,
    WORLD_TICK_PROMPT,
    TASK_TICK_PROMPT,
    EXECUTION_TICK_PROMPT,
    SYSTEM_PROMPT,
)


MAX_BATCH_SIZE = 1000

class LlamaAgent(AgentResponseMixin, BaseAgent):
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
            user_prompt=prompt
        )
        self.logger.info(f"Llama completion: {llama_completion}")
        return LlamaCompletion.from_dict(llama_completion)


    def stream_complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
        return self.complete_text(prompt, max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty, stop, echo, best_of, prompt_tokens, response_format, system_prompt, chat_id, streaming=True)
        
    def complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None, streaming:bool = False):
        # Normalize parameters using shared utility
        params = self._normalize_params(max_tokens, n, temperature, top_p, frequency_penalty, presence_penalty)

        if not system_prompt:
            system_prompt = SYSTEM_PROMPT

        # Build a hydrated prompt that includes chat history if available
        hydrated_user_prompt = prompt
        if chat_id is not None:
            try:
                history = get_chat_history(chat_id)
                history_lines: List[str] = []
                for pair in history.chat_history:
                    user_msg = pair.get("user")
                    ai_msg = pair.get("ai")
                    if user_msg is not None:
                        history_lines.append(f"User: {user_msg}")
                    if ai_msg is not None:
                        history_lines.append(f"Assistant: {ai_msg}")
                history_lines.append(f"User: {prompt}")
                hydrated_user_prompt = "\n".join(history_lines)
            except Exception as e:
                self.logger.warning(f"Failed to hydrate chat history for chat_id={chat_id}: {e}")

        completion = self._complete_llama_sequence(
            prompt=hydrated_user_prompt,
            max_tokens=params.max_tokens,
            system_prompt=system_prompt,
            top_p=params.top_p,
            temperature=params.temperature
        )

        # Use the common extraction method
        ai_message = completion.get_assistant_content()
        chat_history = self._agent_context._add_to_chat_history(user_message=prompt, ai_message=ai_message, chat_id=chat_id)
        completion.chat_id = chat_history.chat_session_id

        return completion
        
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
        self.tick_world()
        self._pop_and_add_execution_tasks()
        self._clear_execution_tasks()

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
