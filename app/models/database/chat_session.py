from sqlalchemy import Integer, Column, String, DateTime, ForeignKey
from app.models.database.database import Base, Session
import json
from typing import List, Dict

class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True)
    # chat history stored as JSON string of message arrays
    chat_history = Column(String)
    create_date = Column(DateTime)
    update_date = Column(DateTime)

def update_chat_history(session_id: int, new_user_message: str, new_ai_message: str):
    '''
    Method to update chat history by ID
    Adds a new user-AI message pair to the conversation
    '''
    session = Session()
    chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()
    
    # Load existing history or create new
    current_history = json.loads(chat_session.chat_history) if chat_session.chat_history else []
    
    # Add new message pair
    current_history.append({
        "user": new_user_message,
        "ai": new_ai_message
    })
    
    # Save updated history
    session.query(ChatSession).filter_by(chat_session_id=session_id).update(
        {"chat_history": json.dumps(current_history)}
    )
    session.commit()
    session.close()

def get_chat_history(session_id: int) -> List[Dict[str, str]]:
    '''
    Method to get chat history by ID
    Returns array of message pairs
    '''
    session = Session()
    chat_session = session.query(ChatSession).filter_by(chat_session_id=session_id).first()
    session.close()
    
    if chat_session and chat_session.chat_history:
        return json.loads(chat_session.chat_history)
    return []