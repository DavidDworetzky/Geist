import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from app.models.database.database import Base, SessionLocal


# Create logger for this module
logger = logging.getLogger(__name__)


class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)

    # chat history stored as JSON string of message arrays
    chat_history = Column(String)
    create_date = Column(DateTime)
    update_date = Column(DateTime)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"))


@dataclass
class ChatHistory(list):
    """
    Data Transfer Object for chat history and session information that behaves like a list

    Attributes:
        chat_history (List[Any]): List of chat messages in any format
        chat_session_id (int): Unique identifier for the chat session
    """

    chat_history: list[Any]
    chat_id: int
    create_date: datetime
    total_messages: int = 0

    def __post_init__(self) -> None:
        """Initialize the list with chat_history contents"""
        super().__init__()
        # Parse JSON string if it's a string, otherwise use as-is
        if isinstance(self.chat_history, str):
            self.chat_history = json.loads(self.chat_history)
        self.extend(self.chat_history)
        self.total_messages = len(self)

    def append(self, item: Any) -> None:
        """
        Append an item to the chat history

        Args:
            item (Any): Item to append to the chat history
        """
        super().append(item)
        self.chat_history = list(self)

    def extend(self, items: list[Any]) -> None:
        """
        Extend the chat history with a list of items

        Args:
            items (List[Any]): Items to extend the chat history with
        """
        super().extend(items)
        self.chat_history = list(self)


def _serialize_chat_extension(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_serialize_chat_extension(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_chat_extension(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_chat_extension(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return value


def update_chat_history(
    new_user_message: str,
    new_ai_message: str | None,
    session_id: int | None = None,
    tool_calls: list[Any] | None = None,
    artifacts: list[Any] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    user_id: int | None = None,
    run_id: str | None = None,
    status: str | None = None,
) -> ChatSession:
    """
    Method to update chat history by ID
    Adds a new user-AI message pair to the conversation
    """
    with SessionLocal() as session:
        chat_session = None
        if session_id:
            chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()

        if not session_id or not chat_session:
            chat_session = ChatSession(
                chat_session_id=session_id,  # Use the provided session_id
                chat_history="[]",
                create_date=datetime.now(),
                update_date=datetime.now(),
                user_id=user_id,
            )
            session.add(chat_session)
        elif user_id is not None and chat_session.user_id is None:
            chat_session.user_id = user_id

        # Load existing history or create new
        current_history = json.loads(chat_session.chat_history) if chat_session.chat_history else []

        # Add new message pair. Extra fields are optional so older chat history
        # records and clients that only read user/ai remain compatible.
        history_entry = {"user": new_user_message, "ai": new_ai_message}
        if tool_calls:
            history_entry["tool_calls"] = _serialize_chat_extension(tool_calls)
        if artifacts:
            history_entry["artifacts"] = _serialize_chat_extension(artifacts)
        if transcript:
            history_entry["transcript"] = _serialize_chat_extension(transcript)
        if run_id:
            history_entry["run_id"] = run_id
        if status:
            history_entry["status"] = status
        current_history.append(history_entry)

        chat_session.chat_history = json.dumps(current_history)
        chat_session.update_date = datetime.now()
        session.commit()
        session.refresh(chat_session)
        logger.info("Updated chat history for session %s", chat_session.chat_session_id)
        return chat_session


def get_chat_history(chat_session_id: int) -> ChatHistory:
    """
    Retrieves chat history for a specific session ID

    Args:
        chat_session_id (int): The ID of the chat session to retrieve

    Returns:
        ChatHistory: DTO containing chat history and session ID
    """
    with SessionLocal() as session:
        chat_session = (
            session.query(ChatSession)
            .filter(ChatSession.chat_session_id == chat_session_id)
            .first()
        )
        if chat_session:
            return ChatHistory(
                chat_history=chat_session.chat_history,
                chat_id=chat_session.chat_session_id,
                create_date=chat_session.create_date,
            )
        # When no chat session exists, create with current timestamp
        return ChatHistory(
            chat_history=[],
            chat_id=chat_session_id,
            create_date=datetime.now(),  # Add current timestamp for new empty histories
        )


def get_paginated_chat_history(
    chat_session_id: int, page: int = 1, page_size: int = 20
) -> ChatHistory:
    """
    Retrieves paginated chat history for a specific session ID.
    Page 1 is the most recent messages.

    Args:
        chat_session_id (int): The ID of the chat session to retrieve
        page (int): Page number (1-indexed, 1 is latest)
        page_size (int): Number of messages per page

    Returns:
        ChatHistory: DTO containing partial chat history and session ID
    """
    with SessionLocal() as session:
        chat_session = (
            session.query(ChatSession)
            .filter(ChatSession.chat_session_id == chat_session_id)
            .first()
        )
        if chat_session:
            full_history_str = chat_session.chat_history
            full_history = json.loads(full_history_str) if full_history_str else []
            total_count = len(full_history)

            # Calculate indices for reverse pagination (latest first)
            end_idx = total_count - ((page - 1) * page_size)
            start_idx = max(0, end_idx - page_size)

            sliced_history = [] if end_idx <= 0 else full_history[start_idx:end_idx]

            return ChatHistory(
                chat_history=sliced_history,
                chat_id=chat_session_id,
                create_date=chat_session.create_date,
                total_messages=total_count,
            )

        # When no chat session exists
        return ChatHistory(
            chat_history=[], chat_id=chat_session_id, create_date=datetime.now(), total_messages=0
        )


def get_all_chat_history() -> list[ChatHistory]:
    """
    Retrieves all chat histories from the database

    Returns:
        List[ChatHistory]: List of ChatHistory DTOs containing chat histories and their session IDs
    """
    with SessionLocal() as session:
        chat_sessions = session.query(ChatSession).all()
        return [
            ChatHistory(
                chat_history=chat_session.chat_history,
                chat_id=chat_session.chat_session_id,
                create_date=chat_session.create_date,
            )
            for chat_session in chat_sessions
        ]


def get_paginated_chat_sessions(page: int = 1, page_size: int = 20) -> list[ChatHistory]:
    """
    Retrieves paginated chat sessions from the database

    Args:
        page (int): Page number (1-indexed)
        page_size (int): Number of sessions per page

    Returns:
        List[ChatHistory]: List of ChatHistory DTOs containing chat histories and their session IDs
    """
    offset = (page - 1) * page_size
    with SessionLocal() as session:
        # Order by create_date desc to show newest sessions first
        chat_sessions = (
            session.query(ChatSession)
            .order_by(ChatSession.create_date.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )

        return [
            ChatHistory(
                chat_history=chat_session.chat_history,
                chat_id=chat_session.chat_session_id,
                create_date=chat_session.create_date,
            )
            for chat_session in chat_sessions
        ]
