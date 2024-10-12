from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Message:
    role: str
    content: str
    refusal: Optional[str] = None

@dataclass
class Choice:
    index: int
    message: Message
    logprobs: Optional[object] = None
    finish_reason: Optional[str] = None

@dataclass
class TokenDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: TokenDetails
    completion_tokens_details: TokenDetails

@dataclass
class Gpt4Completion:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
