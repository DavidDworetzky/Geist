from sqlalchemy import Integer, Column, String, DateTime, ForeignKey
from app.models.database.database import Base, Session, SessionLocal
import json
from typing import List, Dict
from datetime import datetime
import logging

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
        logger.info(f"Updated chat history for session {session_id}, bound obhject session id is : {chat_session.chat_session_id}")
        logger.info(f"Chat History: {chat_session.chat_history}")
        return chat_session 

def get_chat_history(session_id: int, user_id: int = None) -> List[Dict[str, str]]:
    '''
    Method to get chat history by ID
    Returns array of message pairs
    '''
    with SessionLocal() as session:
        chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()
        if chat_session and chat_session.chat_history:
            return json.loads(chat_session.chat_history)
    return []

def get_all_chat_history(user_id: int = None) -> List[Dict[str, str]]:
    '''
    Returns all chat sessions for a user
    '''
    with SessionLocal() as session:
        chat_sessions = session.query(ChatSession).all()
        return [json.loads(chat_session.chat_history) for chat_session in chat_sessions]