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
        )
        except Exception as e:
            raise ValueError(f"Error creating Message from dict: {data}") from e

@dataclass
class LlamaCompletion:
    messages: List[Message]
    chat_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: List[Dict[str, str]]) -> 'LlamaCompletion':
        return cls(
            messages=[Message.from_dict(message) for message in data],
            #chat id is assigned after construction
            chat_id=None
        )
    

def strings_to_message_dict(prompt: str, response: str):
    '''
    Converts autoregressive LLM output into a list of messages
    '''
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return messages