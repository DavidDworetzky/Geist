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
    total_messages: int = 0

    def __post_init__(self) -> None:
        """Initialize the list with chat_history contents"""
        super().__init__()
        # Parse JSON string if it's a string, otherwise use as-is
        if isinstance(self.chat_history, str):
            self.chat_history = json.loads(self.chat_history)
        self.extend(self.chat_history)
        if self.total_messages == 0:
            self.total_messages = len(self)

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
            chat_session = ChatSession(
                chat_session_id=session_id,  # Use the provided session_id
                chat_history="[]",
                create_date=datetime.now(),
                update_date=datetime.now()
            )
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
                create_date=chat_session.create_date
            )
        # When no chat session exists, create with current timestamp
        return ChatHistory(
            chat_history=[],
            chat_id=chat_session_id,
            create_date=datetime.now()  # Add current timestamp for new empty histories
        )

def get_paginated_chat_history(chat_session_id: int, page: int = 1, page_size: int = 20) -> ChatHistory:
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
        chat_session = session.query(ChatSession).filter(ChatSession.chat_session_id == chat_session_id).first()
        if chat_session:
            full_history_str = chat_session.chat_history
            full_history = json.loads(full_history_str) if full_history_str else []
            total_count = len(full_history)
            
            # Calculate indices for reverse pagination (latest first)
            # Page 1: last 20 items -> indices [total-20 : total]
            # Page 2: items before that -> indices [total-40 : total-20]
            end_idx = total_count - ((page - 1) * page_size)
            start_idx = max(0, end_idx - page_size)
            
            if end_idx <= 0:
                sliced_history = []
            else:
                # Python slicing: [start:end]. 
                # If end_idx > total_count, it's fine (capped at len).
                # If start_idx < 0, it's 0.
                # If end_idx is reduced to match array bounds? 
                # end_idx depends on page. 
                # Example: total=100, page=1, size=20. end=100, start=80. slice [80:100] (20 items)
                # Example: total=100, page=5, size=20. end=20, start=0. slice [0:20]
                # Example: total=100, page=6, size=20. end=0. slice [0:0] -> empty. Correct.
                sliced_history = full_history[start_idx:end_idx]
                
            return ChatHistory(
                chat_history=sliced_history,
                chat_id=chat_session_id,
                create_date=chat_session.create_date,
                total_messages=total_count
            )
            
        # When no chat session exists
        return ChatHistory(
            chat_history=[],
            chat_id=chat_session_id,
            create_date=datetime.now(),
            total_messages=0
        )

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

def get_paginated_chat_sessions(page: int = 1, page_size: int = 20) -> List[ChatHistory]:
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
        chat_sessions = session.query(ChatSession)\
            .order_by(ChatSession.create_date.desc())\
            .offset(offset)\
            .limit(page_size)\
            .all()
            
        return [
            ChatHistory(
                chat_history=chat_session.chat_history,
                chat_id=chat_session.chat_session_id,
                create_date=chat_session.create_date
            ) 
            for chat_session in chat_sessions
        ]