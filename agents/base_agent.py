from abc import ABC, abstractmethod
from agents.agent_context import AgentContext
import logging

class BaseAgent(ABC):

    def __init__(self, agent_context: AgentContext, as_subprocess = False):
        self._agent_context = agent_context
        self.as_subprocess = as_subprocess

    @abstractmethod
    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text", system_prompt:str = None, chat_id:int = None):
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
        Restart an agent's subprocess and rehydrate it (phase in, heh)
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
