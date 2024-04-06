from abc import ABC, abstractmethod
from agents.agent_context import AgentContext

class BaseAgent(ABC):

    def __init__(self, agent_context: AgentContext):
        self.agent_context = agent_context

    @abstractmethod
    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
        pass

    @abstractmethod
    def initialize(self, task_prompt):
        pass

    @abstractmethod
    def freeze(self):
        '''
        Freeze the agent's state and save it to context.
        '''
        pass

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
