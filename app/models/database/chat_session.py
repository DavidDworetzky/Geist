from sqlalchemy import Integer, Column, String, DateTime, ForeignKey
from app.models.database.database import Base, Session, SessionLocal
import json
from typing import List, Dict
from datetime import datetime

class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True)
    # chat history stored as JSON string of message arrays
    chat_history = Column(String)
    create_date = Column(DateTime)
    update_date = Column(DateTime)
    user_id = Column(Integer, ForeignKey('user.user_id'))

def update_chat_history(session_id: int, new_user_message: str, new_ai_message: str):
    '''
    Method to update chat history by ID
    Adds a new user-AI message pair to the conversation
    '''
    with SessionLocal() as session:
        chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()

        if not chat_session:
            chat_session = ChatSession(chat_history="[]", create_date=datetime.now(), update_date=datetime.now())
        
        # Load existing history or create new
        current_history = json.loads(chat_session.chat_history) if chat_session.chat_history else []
        
        # Add new message pair
        current_history.append({
            "user": new_user_message,
            "ai": new_ai_message
        })

        chat_session.chat_history = json.dumps(current_history)
        session.commit()

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