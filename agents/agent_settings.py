#import dataclass
from dataclasses import dataclass

@dataclass
class AgentSettings:
    name: str
    version: str
    description: str
    #conversational completion parameters for conversation with the agent.
    