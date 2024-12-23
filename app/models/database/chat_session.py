from sqlalchemy import Integer, Column, String, DateTime, ForeignKey
from app.models.database.database import Base, Session, SessionLocal
import json
from typing import List, Dict, Any
from datetime import datetime
import logging
from dataclasses import dataclass

# Create logger for this module
logger = logging.getLogger(__name__)

class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # chat history stored as JSON string of message arrays
    chat_history = Column(String)
    create_date = Column(DateTime)
    update_date = Column(DateTime)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'))

@dataclass
class ChatHistory(list):
    """
    Data Transfer Object for chat history and session information that behaves like a list
    
    Attributes:
        chat_history (List[Any]): List of chat messages in any format
        chat_session_id (int): Unique identifier for the chat session
    """
    chat_history: List[Any]
    chat_id: int
    create_date: datetime

    def __post_init__(self) -> None:
        """Initialize the list with chat_history contents"""
        super().__init__()
        # Parse JSON string if it's a string, otherwise use as-is
        if isinstance(self.chat_history, str):
            self.chat_history = json.loads(self.chat_history)
        self.extend(self.chat_history)

    def append(self, item: Any) -> None:
        """
        Append an item to the chat history
        
        Args:
            item (Any): Item to append to the chat history
        """
        super().append(item)
        self.chat_history = list(self)

    def extend(self, items: List[Any]) -> None:
        """
        Extend the chat history with a list of items
        
        Args:
            items (List[Any]): Items to extend the chat history with
        """
        super().extend(items)
        self.chat_history = list(self)

def update_chat_history(new_user_message: str, new_ai_message: str, session_id: int = None) -> ChatSession:
    '''
    Method to update chat history by ID
    Adds a new user-AI message pair to the conversation
    '''
    with SessionLocal() as session:
        if session_id:
            chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()

        if not session_id or not chat_session:
            chat_session = ChatSession(chat_history="[]", create_date=datetime.now(), update_date=datetime.now())
            session.add(chat_session)
        
        # Load existing history or create new
        current_history = json.loads(chat_session.chat_history) if chat_session.chat_history else []
        
        # Add new message pair
        current_history.append({
            "user": new_user_message,
            "ai": new_ai_message
        })

        chat_session.chat_history = json.dumps(current_history)
        session.commit()
        logger.info(f"Updated chat history for session {session_id}, bound object session id is : {chat_session.chat_session_id}")
        logger.info(f"Chat History: {chat_session.chat_history}")
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
        chat_session = session.query(ChatSession).filter(ChatSession.chat_session_id == chat_session_id).first()
        if chat_session:
            return ChatHistory(
                chat_history=chat_session.chat_history,
                chat_id=chat_session.chat_session_id,
                create_date= chat_session.create_date
            )
        return ChatHistory(chat_history=[], chat_id=chat_session_id)

def get_all_chat_history() -> List[ChatHistory]:
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
                create_date=chat_session.create_date
            ) 
            for chat_session in chat_sessions
        ]