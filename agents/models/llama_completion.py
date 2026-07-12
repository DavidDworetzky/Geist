from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str
    chat_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> 'Message':
        try:
            return cls(
            role=data.get('role', ''),
            content=data.get('content', ''),
        )
        except Exception as e:
            raise ValueError(f"Error creating Message from dict: {data}") from e

@dataclass
class LlamaCompletion:
    messages: list[Message]
    chat_id: int | None = None

    @classmethod
    def from_dict(cls, data: list[dict[str, str]]) -> 'LlamaCompletion':
        return cls(
            messages=[Message.from_dict(message) for message in data],
            #chat id is assigned after construction
            chat_id=None
        )

    def get_assistant_content(self) -> str | None:
        """The first assistant message's content, or None if there is none."""
        return next((message.content for message in self.messages if message.role == 'assistant'), None)


def strings_to_message_dict(prompt: str, response: str) -> list[dict[str, str]]:
    '''
    Converts autoregressive LLM output into a list of messages
    '''
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return messages
