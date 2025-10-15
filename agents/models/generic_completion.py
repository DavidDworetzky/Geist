from dataclasses import dataclass
from typing import List, Optional, Dict
import json

@dataclass
class Message:
    role: str
    content: str
    refusal: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

@dataclass
class Choice:
    index: int
    message: Message
    logprobs: Optional[object] = None
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict):
        # If data is already a Choice object, return it
        if isinstance(data, cls):
            return data
        # If message is already a Message object, use it directly
        if isinstance(data['message'], Message):
            message = data['message']
        else:
            message = Message.from_dict(data['message'])

        # Create a copy of data to avoid modifying the original
        data_copy = data.copy()
        data_copy['message'] = message
        return cls(**data_copy)

@dataclass
class TokenDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict):
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: TokenDetails
    completion_tokens_details: TokenDetails

    @classmethod
    def from_dict(cls, data: Dict):
        data['prompt_tokens_details'] = TokenDetails.from_dict(data['prompt_tokens_details'])
        data['completion_tokens_details'] = TokenDetails.from_dict(data['completion_tokens_details'])
        return cls(**data)

@dataclass
class GenericCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
    chat_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict):
        data['choices'] = [Choice.from_dict(choice) for choice in data['choices']]
        data['usage'] = Usage.from_dict(data['usage'])
        return cls(**data)
