from abc import ABC, abstractmethod
from agents.agent_context import AgentContext

class BaseAgent(ABC):

    def __init__(self, agent_context: AgentContext):
        self._agent_context = agent_context

    @abstractmethod
    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
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

    def phase_in(self):
        '''
        Restart an agent's subprocess and rehydrate it (phase in, heh)
        '''

    @abstractmethod
    def tick(self):
        self.tick_world()
        self.tick_tasks()
        self.tick_execution()


    @abstractmethod
    def tick_world(self):
        pass

    @abstractmethod
    def tick_tasks(self):
        pass
    
    @abstractmethod
    def tick_execution(self):
        pass
