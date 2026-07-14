from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str
    refusal: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

@dataclass
class Choice:
    index: int
    message: Message
    logprobs: object | None = None
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Choice':
        # If data is already a Choice object, return it
        if isinstance(data, cls):
            return data
        # If message is already a Message object, use it directly
        if isinstance(data['message'], Message):
            message = data['message']
        else:
            message = Message.from_dict(data['message'])

        # Filter to only valid fields and set message
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        valid_fields['message'] = message
        return cls(**valid_fields)

@dataclass
class TokenDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> 'TokenDetails':
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
    def from_dict(cls, data: dict) -> 'Usage':
        # If data is already a Usage object, return it
        if isinstance(data, cls):
            return data

        # Filter to only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        # Convert nested objects if they're not already the correct type
        if 'prompt_tokens_details' in valid_fields and not isinstance(valid_fields['prompt_tokens_details'], TokenDetails):
            valid_fields['prompt_tokens_details'] = TokenDetails.from_dict(valid_fields['prompt_tokens_details'])

        if 'completion_tokens_details' in valid_fields and not isinstance(valid_fields['completion_tokens_details'], TokenDetails):
            valid_fields['completion_tokens_details'] = TokenDetails.from_dict(valid_fields['completion_tokens_details'])

        return cls(**valid_fields)

@dataclass
class GenericCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None
    chat_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> 'GenericCompletion':
        # Filter to only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        valid_fields['choices'] = [Choice.from_dict(choice) for choice in data['choices']]
        valid_fields['usage'] = Usage.from_dict(data['usage'])
        return cls(**valid_fields)

    def get_assistant_content(self) -> str | None:
        """The first choice's assistant message content, or None if empty."""
        if self.choices:
            return self.choices[0].message.content
        return None
