from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Message:
    role: str
    content: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Message':
        try: 
            return cls(
            role=data.get('role', ''),
            content=data.get('content', '')
        )
        except Exception as e:
            raise ValueError(f"Error creating Message from dict: {data}") from e

@dataclass
class LlamaCompletion:
    messages: List[Message]

    @classmethod
    def from_dict(cls, data: List[Dict[str, str]]) -> 'LlamaCompletion':
        return cls(
            messages=[Message.from_dict(message) for message in data]
        )
