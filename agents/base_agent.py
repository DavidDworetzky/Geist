from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def tick(self):
        pass
