#import dataclass
from dataclasses import dataclass

@dataclass
class AgentSettings:
    name: str
    version: str
    description: str
    #conversational completion parameters for conversation with the agent.
    max_tokens: int = 16
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    echo: bool = False
