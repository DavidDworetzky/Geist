from abc import ABC, abstractmethod
from agents.agent_context import AgentContext
import logging
from typing import Iterator, Optional

class BaseAgent(ABC):

    def __init__(self, agent_context: AgentContext, as_subprocess = False):
        self._agent_context = agent_context
        self.as_subprocess = as_subprocess

    @abstractmethod
    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
        pass
    
    @abstractmethod
    def stream_complete_text(self, prompt: str, max_tokens: int = 16, n: int = 1, stop: Optional[str] = None, temperature: float = 1.0, top_p: float = 1, frequency_penalty: float = 0, presence_penalty: float = 0, echo: bool = False, best_of: Optional[int] = None, prompt_tokens: Optional[int] = None, response_format: str = "text", system_prompt: Optional[str] = None, chat_id: Optional[int] = None) -> Iterator[str]:
        """
        Stream text completion token by token or in chunks.
        
        This is an optional method that agents can override to provide streaming support.
        By default, it raises NotImplementedError.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            n: Number of completions (typically 1 for streaming)
            stop: Stop sequences
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            echo: Whether to echo the prompt
            best_of: Number of completions to generate server-side
            prompt_tokens: Number of prompt tokens
            response_format: Response format
            system_prompt: System prompt
            chat_id: Chat session ID
            
        Yields:
            str: Text chunks/tokens as they are generated
            
        Raises:
            NotImplementedError: If streaming is not supported by this agent
        """
        pass

    @abstractmethod
    def complete_audio(self, audio_file, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
        pass

    @abstractmethod
    def connect_realtime_audio(self):
        pass

    @abstractmethod
    def initialize(self, task_prompt):
        pass

    @abstractmethod
    def phase_out(self):
        '''
        Phase out an agent's state and stop it (aka, stop subprocess, save state)
        '''
        pass

    @abstractmethod
    def phase_in(self):
        '''
        Restart an agent's subprocess and rehydrate it (phase in)
        '''

    @abstractmethod
    def tick(self):
        if self._agent_context.settings.include_world_processing:
            self.tick_world()
        self.tick_tasks()
        self.tick_execution()

    def state(self):
        logging.info("getting agent state.")
        return {
            "world_context" : self._agent_context.world_context,
            "task_context" : self._agent_context.task_context,
            "execution_context" : self._agent_context.execution_context
        }


    @abstractmethod
    def tick_world(self):
        pass

    @abstractmethod
    def tick_tasks(self):
        pass
    
    @abstractmethod
    def tick_execution(self):
        pass
