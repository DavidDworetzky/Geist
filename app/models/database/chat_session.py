from sqlalchemy import Integer, Column, String, DateTime, ForeignKey
from app.models.database.database import Base, Session

class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True)
    #chat history is handled as one continuous string
    chat_history = Column(String)
    create_date = Column(DateTime)
    update_date = Column(DateTime)


#method for updating chat history, given a session id and a new chat message
def update_chat_history(session_id: int, new_message: str):
    '''
    Method to update chat history by ID
    '''
    session = Session()
    session.query(ChatSession).filter_by(chat_session_id=session_id).update({"chat_history": new_message})
    session.commit()
    session.close()

def get_chat_history(session_id: int):
    '''
    Method to get chat history by ID
    '''
    session = Session()
    chat_history = session.query(ChatSession).filter_by(chat_session_id=session_id).first()
    session.close()
    return chat_history