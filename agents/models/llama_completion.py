from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Message:
    role: str
    content: str
    chat_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Message':
        try: 
            return cls(
            role=data.get('role', ''),
            content=data.get('content', ''),
            chat_id=data.get('chat_id', None)
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
