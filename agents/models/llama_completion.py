from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class LlamaCompletion:
    id: str
    message: str