#using dataclass
from dataclasses import dataclass
from typing import List, Optional
from app.models.restriction import Restriction

@dataclass
class Agent:
    #language model presets
    name : str
    version : str
    description : str
    max_tokens : int
    n : int
    temperature : int
    top_p : int
    frequency_penalty : int
    presence_penalty : int
    tags : str
    #memory presets
    working_context_length : int
    long_term_context_length : int
    #prompt presets
    prompt : str
    restrictions : List[Restriction]