from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AgentCompletion:
    message: List[str]
    id: str